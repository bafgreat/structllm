"""Self-contained fine-tuning files from accepted teacher trajectories.

The export needs nothing from StructSub at training time: each line carries
the conversation in the message shape Hugging Face chat templates consume
(``reasoning_content``, ``tool_calls`` with object arguments, ``tool``
messages) and the function definitions that were offered, so a tokenizer's
own template can render it. Assistant messages keep a ``train`` flag; loss is
taken only on assistant turns with ``train: true``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from structllm.chat_format import to_template_messages
from structllm.dataset import canonical_json
from structllm.datagen.grounding import ungrounded

EXPORT_SCHEMA_VERSION = "1.0.0"


def template_example(record: dict[str, Any]) -> dict[str, Any]:
    converted = to_template_messages(record["messages"])
    messages = []
    for original, message in zip(record["messages"], converted):
        if original["role"] == "assistant":
            message = {**message, "train": bool(original.get("train", True))}
        messages.append(message)
    return {
        "id": record["record_id"],
        "messages": messages,
        "tools": record["tools"],
        "meta": {
            "request_id": record["request_id"],
            "scenario_id": record["scenario_id"],
            "behaviour": record["behaviour"],
            "turn_behaviours": [turn["behaviour"] for turn in record.get("turns", [])],
            "executed_calls": record["executed_calls"],
            "mocked_calls": record["mocked_calls"],
            "catalogue_sha256": record["catalogue_sha256"],
            "teacher": record["teacher"]["adapter"],
        },
    }


def salvage_prefix(record: dict[str, Any]) -> dict[str, Any] | None:
    """The turns of a rejected conversation that passed, as a shorter conversation.

    A conversation is rejected as a whole when any turn fails, but every turn before
    the first failure was checked and passed. They are a valid shorter conversation,
    cut just before the user message that started the failing turn, with only the
    tools that had been offered by then. Nothing after the failure is kept.
    """
    turns = record.get("turns") or []
    first_bad = next((index for index, turn in enumerate(turns) if turn.get("problems")), None)
    users = [index for index, message in enumerate(record["messages"]) if message["role"] == "user"]
    if not first_bad or first_bad >= len(users):
        return None  # nothing passed before the failure, or no later turn to cut at
    kept = record["messages"][: users[first_bad]]
    wanted = set().union(*map(set, record.get("tools_by_turn", [])[:first_bad]))
    for message in kept:
        for call in message.get("tool_calls") or []:
            wanted.add((call.get("function") or call).get("name"))
    tools = [tool for tool in record["tools"] if (tool.get("function") or tool).get("name") in wanted]
    return {
        **record,
        "messages": kept,
        "turns": turns[:first_bad],
        "tools": tools,
        "tools_by_turn": record.get("tools_by_turn", [])[:first_bad],
        "record_id": f"{record['record_id']}.prefix{first_bad}",
        "accepted": True,
        "salvaged_turns": first_bad,
    }


def collect(root: Path, *, salvage: bool = False) -> list[dict[str, Any]]:
    """Accepted records from a trajectory directory and any shard directories below it.

    With ``salvage``, the passing opening turns of rejected conversations are added too
    (see :func:`salvage_prefix`); an accepted record for the same request always wins.
    """
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("accepted.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    records.setdefault(record["request_id"], record)
    if salvage:
        for path in sorted(root.rglob("rejected.jsonl")):
            with path.open(encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        record = json.loads(line)
                        prefix = salvage_prefix(record)
                        if prefix is not None:
                            records.setdefault(record["request_id"], prefix)
    return [records[key] for key in sorted(records)]


def export_sft(root: Path, output_dir: Path, *, require_grounded: bool = True, salvage: bool = True) -> Path:
    records = collect(root, salvage=salvage)
    salvaged = sum(1 for record in records if record.get("salvaged_turns"))
    if salvaged:
        print(f"Added the passing opening turns of {salvaged} rejected conversations")
    ungrounded_ids: dict[str, list[str]] = {}
    if require_grounded:
        kept = []
        for record in records:
            problems = ungrounded(record["messages"], record["tools"])
            if problems:
                ungrounded_ids[record["record_id"]] = [number for _, numbers in problems for number in numbers]
            else:
                kept.append(record)
        if ungrounded_ids:
            print(f"Dropped {len(ungrounded_ids)} of {len(records)} conversations that state numbers no tool result supports")
        records = kept
    if not records:
        raise SystemExit(f"No accepted trajectories under {root}")
    catalogues = {record["catalogue_sha256"] for record in records}
    if len(catalogues) != 1:
        raise SystemExit(f"Trajectories come from {len(catalogues)} different catalogues; export them separately")
    examples = {"train": [], "eval": []}
    for record in records:
        examples[record["split"]].append(template_example(record))
    digest = hashlib.sha256(canonical_json(sorted(record["record_id"] for record in records))).hexdigest()[:12]
    destination = output_dir / f"structllm-sft-{next(iter(catalogues))[:12]}-{digest}"
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, items in examples.items():
        path = destination / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as file:
            for item in items:
                file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def tally(items: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            counts[item["meta"][key]] = counts.get(item["meta"][key], 0) + 1
        return dict(sorted(counts.items()))

    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "catalogue_sha256": next(iter(catalogues)),
        "counts": {split: len(items) for split, items in examples.items()},
        "scenarios": {split: tally(items, "scenario_id") for split, items in examples.items()},
        "behaviours": {split: tally(items, "behaviour") for split, items in examples.items()},
        "assistant_turns": {
            split: {
                "trained": sum(1 for item in items for m in item["messages"] if m["role"] == "assistant" and m["train"]),
                "context_only": sum(1 for item in items for m in item["messages"] if m["role"] == "assistant" and not m["train"]),
            }
            for split, items in examples.items()
        },
        "tool_results": {
            "executed": sum(record["executed_calls"] for record in records),
            "mocked": sum(record["mocked_calls"] for record in records),
        },
        "salvaged_prefixes": salvaged,
        "grounding": {
            "required": require_grounded,
            "dropped": len(ungrounded_ids),
            "dropped_records": dict(sorted(ungrounded_ids.items())),
        },
        "files": files,
        "format": (
            "One JSON object per line: messages (Hugging Face chat-template shape; assistant "
            "messages carry train=true/false), tools (function definitions offered in the "
            "conversation), meta. Splits are by whole scenario."
        ),
        "limitations": [
            "Accepted means the conversation had the expected shape. Numbers in assistant text were checked against tool results, the user's messages and the tool definitions (see grounding); other scientific statements were not verified.",
            "Mocked results are locally created awaiting_approval envelopes; those operations never ran.",
            "Executed results come from a StructSub development server.",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def assemble_sft(exports: list[Path], output_dir: Path, *, system_prompt: str | None = None) -> Path:
    """Join several exports (for example one per catalogue) into one training set.

    Every row keeps its own ``tools`` list, so exports from different catalogues can be
    concatenated. The system prompt is rewritten to the current one (keeping any
    ``Active project_id`` suffix), so the model is trained on the prompt it will be served
    with. A scenario that landed in both splits is kept in train only, so eval is never
    trained on, and row ids must be unique.
    """
    from structllm.orchestrator import SYSTEM_PROMPT

    prompt = system_prompt or SYSTEM_PROMPT
    marker = "\n\nActive project_id"
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for export in exports:
        for split in rows:
            with (export / f"{split}.jsonl").open(encoding="utf-8") as file:
                rows[split].extend(json.loads(line) for line in file if line.strip())
    for split_rows in rows.values():
        for row in split_rows:
            first = row["messages"][0]
            if first["role"] == "system":
                content = first["content"]
                first["content"] = prompt + (content[content.index(marker):] if marker in content else "")
    trained = {row["meta"]["scenario_id"] for row in rows["train"]}
    leaked = {row["meta"]["scenario_id"] for row in rows["eval"]} & trained
    rows["eval"] = [row for row in rows["eval"] if row["meta"]["scenario_id"] not in trained]
    ids = [row["id"] for split_rows in rows.values() for row in split_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate conversation ids across the exports")
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, split_rows in rows.items():
        with (output_dir / f"{split}.jsonl").open("x", encoding="utf-8") as file:
            for row in split_rows:
                file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "sources": [export.name for export in exports],
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "scenarios": len({row["meta"]["scenario_id"] for split_rows in rows.values() for row in split_rows}),
        "system_prompt_normalised": True,
        "eval_rows_removed_for_scenario_overlap": len(leaked),
        "note": "Concatenation; each row carries its own tools list.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir
