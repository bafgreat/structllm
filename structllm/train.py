"""LoRA / QLoRA fine-tuning on the exported SFT conversations.

Each assistant message marked ``train: true`` becomes one example: the prompt is
every message before it (rendered with the model's own chat template, with the
tools offered in that conversation) and the target is that one turn. The loss is
taken on the target only. This keeps earlier reasoning visible, because Qwen3's
template drops ``<think>`` blocks from turns before the latest user message.

Evaluation runs inside the training loop, at the end of every epoch or every
``--eval-every`` optimiser steps:

* eval loss on every held-out example, and
* behaviour on a fixed sample of held-out turns: the model generates the turn
  greedily and it is scored for the right decision (call or answer), the right
  tools, schema-valid and exact arguments, grounded numbers (``structllm.behaviour``).

Each evaluation is appended to ``metrics.jsonl``. The adapter is checkpointed to
``last/`` (with optimiser state, so ``--resume`` continues) and the best by
behaviour score, then eval loss, to ``best/``. ``--patience`` stops a run that
has stopped improving. Neither evaluation needs a StructSub server.

Data preparation needs only a tokenizer. ``--prepare-only`` stops after it and
prints length statistics, so the data can be checked without a GPU.

    uv run structllm-train --data <sft dir> --model Qwen/Qwen3-8B --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    source: str  # "<request_id>#<message index>"
    prompt_len: int = 0
    tools: list[dict[str, Any]] = field(default_factory=list)
    prefix: list[dict[str, Any]] = field(default_factory=list)  # messages before the target turn
    expected: list[dict[str, Any]] = field(default_factory=list)  # the teacher's tool calls in the target turn


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the loss flag, which is not a chat-template field."""
    return [{key: value for key, value in message.items() if key != "train"} for message in messages]


def render_pair(tokenizer: Any, record: dict[str, Any], index: int, *, enable_thinking: bool = True) -> tuple[str, str] | None:
    """(prompt, target) text for assistant message ``index``, or None if the template is not prefix-stable."""
    messages = _template_messages(record["messages"])
    tools = record.get("tools") or None
    options = {"tools": tools, "tokenize": False, "enable_thinking": enable_thinking}
    prompt = tokenizer.apply_chat_template(messages[:index], add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(messages[: index + 1], add_generation_prompt=False, **options)
    if not full.startswith(prompt):
        return None
    target = full[len(prompt):]
    return (prompt, target) if target.strip() else None


def build_examples(tokenizer: Any, records: list[dict[str, Any]], *, max_length: int) -> tuple[list[Example], dict[str, int]]:
    """One masked example per trained assistant turn."""
    from structllm.behaviour import expected_from_message

    stats = {"turns": 0, "kept": 0, "too_long": 0, "not_prefix_stable": 0}
    examples: list[Example] = []
    for record in records:
        for index, message in enumerate(record["messages"]):
            if message["role"] != "assistant" or not message.get("train", False):
                continue
            stats["turns"] += 1
            pair = render_pair(tokenizer, record, index)
            if pair is None:
                stats["not_prefix_stable"] += 1
                continue
            prompt_ids = tokenizer(pair[0], add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(pair[1], add_special_tokens=False)["input_ids"]
            if len(prompt_ids) + len(target_ids) > max_length:
                stats["too_long"] += 1
                continue
            examples.append(
                Example(
                    input_ids=prompt_ids + target_ids,
                    labels=[IGNORE_INDEX] * len(prompt_ids) + target_ids,
                    source=f"{record['meta']['request_id']}#{index}",
                    prompt_len=len(prompt_ids),
                    tools=record.get("tools") or [],
                    prefix=_template_messages(record["messages"][:index]),
                    expected=expected_from_message(message),
                )
            )
            stats["kept"] += 1
    return examples, stats


def describe(examples: list[Example]) -> str:
    if not examples:
        return "no examples"
    lengths = sorted(len(example.input_ids) for example in examples)
    targets = [sum(1 for label in example.labels if label != IGNORE_INDEX) for example in examples]
    return (
        f"{len(examples)} examples; tokens per example median {lengths[len(lengths) // 2]}, "
        f"max {lengths[-1]}; trained tokens per example median {sorted(targets)[len(targets) // 2]}, "
        f"total {sum(targets)}"
    )


def behaviour_sample(examples: list[Example], count: int, seed: int = 0) -> list[Example]:
    """The same turns every time, so scores are comparable between evaluations."""
    if count <= 0 or not examples:
        return []
    return random.Random(seed).sample(examples, min(count, len(examples)))


def better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """Higher behaviour score wins; eval loss breaks ties."""
    if best is None:
        return True
    if candidate["score"] != best["score"]:
        return candidate["score"] > best["score"]
    return candidate["eval_loss"] < best["eval_loss"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA / QLoRA fine-tuning on exported StructLLM conversations.")
    parser.add_argument("--data", type=Path, required=True, help="SFT directory with train.jsonl and eval.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action="store_true", help="4-bit base weights (needs the qlora extra, CUDA only)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N optimiser steps (0: at the end of each epoch)")
    parser.add_argument("--behaviour-samples", type=int, default=48, help="Held-out turns to generate and score at each evaluation (0 turns it off)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tool-format", default="hermes", choices=["hermes", "qwen3_xml"], help="How the model writes tool calls (Qwen3: hermes)")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many evaluations without improvement (0: never)")
    parser.add_argument("--resume", action="store_true", help="Continue from output-dir/last if it exists")
    parser.add_argument("--prepare-only", action="store_true", help="Tokenize, print statistics and stop")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_examples, train_stats = build_examples(tokenizer, read_jsonl(args.data / "train.jsonl"), max_length=args.max_length)
    eval_examples, eval_stats = build_examples(tokenizer, read_jsonl(args.data / "eval.jsonl"), max_length=args.max_length)
    print(f"train: {describe(train_examples)}; {train_stats}")
    print(f"eval:  {describe(eval_examples)}; {eval_stats}")
    if train_stats["not_prefix_stable"] or eval_stats["not_prefix_stable"]:
        print("warning: some turns were skipped because the chat template does not render them as a stable prefix")
    if args.prepare_only:
        return
    if not train_examples:
        raise SystemExit("No training examples.")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

    from structllm.behaviour import aggregate, score_turn
    from structllm.chat_format import parse_generation

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    last_dir, best_dir = args.output_dir / "last", args.output_dir / "best"

    load_options: dict[str, Any] = {"dtype": torch.bfloat16 if args.dtype == "bfloat16" else torch.float32}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
    placement = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=placement, **load_options)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    def loss_of(example: Example) -> Any:
        """Mean cross-entropy over the target turn.

        The target is the end of the sequence, so only the logits for those positions
        are computed (``logits_to_keep``). The full-sequence logits are length x vocabulary
        (about 150k), which is what runs a long example out of memory, not the model.
        """
        ids = torch.tensor([example.input_ids], device=device)
        targets = len(example.input_ids) - example.prompt_len
        logits = model(input_ids=ids, logits_to_keep=targets + 1).logits[0, :-1].float()
        return torch.nn.functional.cross_entropy(logits, ids[0, -targets:])

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_examples) / args.grad_accum)
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    schedule = get_cosine_schedule_with_warmup(optimiser, max(1, total_steps // 20), total_steps)

    step = 0
    epoch = 0
    best: dict[str, Any] | None = None
    bad_evaluations = 0
    if args.resume and (last_dir / "state.pt").exists():
        state = torch.load(last_dir / "state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(last_dir / "adapter_model.safetensors")))
        optimiser.load_state_dict(state["optimiser"])
        schedule.load_state_dict(state["schedule"])
        step, epoch, best, bad_evaluations = state["step"], state["epoch"], state["best"], state["bad_evaluations"]
        random.setstate(state["python_rng"])
        print(f"resumed from step {step}, epoch {epoch}")

    sample = behaviour_sample(eval_examples, args.behaviour_samples)

    def generate(example: Example) -> str:
        ids = torch.tensor([example.input_ids[: example.prompt_len]], device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def evaluate() -> dict[str, Any]:
        """Eval loss on all held-out turns, behaviour on a fixed sample. Leaves the model in train mode."""
        model.eval()
        result: dict[str, Any] = {"step": step, "epoch": epoch, "eval_loss": None, "score": 0.0}
        with torch.no_grad():
            losses = [loss_of(example).item() for example in eval_examples]
        result["eval_loss"] = round(sum(losses) / len(losses), 5) if losses else float("inf")
        if sample:
            scores = []
            for example in sample:
                parsed = parse_generation(generate(example), args.tool_format)
                scores.append(score_turn(parsed, example.expected, example.tools, example.prefix))
            behaviour = aggregate(scores)
            result["behaviour"] = behaviour
            result["score"] = behaviour["score"]
        model.train()
        return result

    def checkpoint(result: dict[str, Any]) -> None:
        nonlocal best, bad_evaluations
        model.save_pretrained(last_dir)
        improved = better(result, best)
        if improved:
            best, bad_evaluations = result, 0
            model.save_pretrained(best_dir)
        else:
            bad_evaluations += 1
        torch.save(
            {"optimiser": optimiser.state_dict(), "schedule": schedule.state_dict(), "step": step, "epoch": epoch,
             "best": best, "bad_evaluations": bad_evaluations, "python_rng": random.getstate()},
            last_dir / "state.pt",
        )
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({**result, "best": improved}) + "\n")
        behaviour = result.get("behaviour") or {}
        summary = ", ".join(f"{key} {behaviour[key]}" for key in ("decision", "names", "args_valid", "args_exact", "grounded") if behaviour.get(key) is not None)
        print(f"eval @ step {step} epoch {epoch}: loss {result['eval_loss']}, score {result['score']} ({summary})"
              f"{'  <- best' if improved else ''}", flush=True)

    model.train()
    pending = 0
    done = step >= total_steps
    stop_early = False
    while not done:
        order = list(range(len(train_examples)))
        random.shuffle(order)
        for position in order:
            (loss_of(train_examples[position]) / args.grad_accum).backward()
            pending += 1
            if pending == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimiser.step()
                schedule.step()
                optimiser.zero_grad(set_to_none=True)
                pending = 0
                step += 1
                if step % 5 == 0:
                    print(f"step {step}/{total_steps} lr {schedule.get_last_lr()[0]:.2e}", flush=True)
                if args.eval_every and step % args.eval_every == 0 and step < total_steps:
                    checkpoint(evaluate())
                    stop_early = bool(args.patience) and bad_evaluations >= args.patience
                if step >= total_steps or stop_early:
                    done = True
                    break
        epoch += 1
        if not args.eval_every or done:
            checkpoint(evaluate())
            stop_early = stop_early or (bool(args.patience) and bad_evaluations >= args.patience)
            done = done or stop_early
    if stop_early:
        print(f"stopped early: no improvement in {args.patience} evaluations")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training.json").write_text(
        json.dumps({"base_model": args.model, "data": str(args.data), "args": {k: str(v) for k, v in vars(args).items()},
                    "train": train_stats, "eval": eval_stats, "steps": step, "stopped_early": stop_early,
                    "best": best, "best_adapter": str(best_dir) if best else None}, indent=2),
        encoding="utf-8",
    )
    if best:
        tokenizer.save_pretrained(best_dir)
        print(f"best evaluation: step {best['step']}, score {best['score']}, loss {best['eval_loss']} -> {best_dir}")
    print(f"Saved final adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
