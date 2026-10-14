"""Replaceable model back ends behind one small interface.

The orchestrator calls ``adapter.next_turn(messages, tools)`` with internal
messages and model-facing tool definitions and receives a ``ParsedTurn``. An
adapter never executes anything: it only proposes text and tool calls.

* ``ScriptedAdapter``: deterministic turns for tests and fixtures.
* ``AnthropicAdapter``: Claude via the official SDK (teacher / reference runs).
* ``OpenAICompatibleAdapter``: any OpenAI-compatible chat endpoint (vLLM,
  SGLang, llama.cpp server) - the usual way to serve the fine-tuned model.
* ``HuggingFaceAdapter``: local ``transformers`` generation with the model's
  native chat template (offline evaluation on a GPU node).

No provider, endpoint or credential is required by the rest of the package.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from structllm.chat_format import ParsedTurn, parse_generation, to_template_messages
from structllm.toolspec import ToolView

_REASONING_TAG = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0

    def add(self, other: Usage) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "requests"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class ModelAdapter(Protocol):
    name: str

    def next_turn(self, messages: list[dict[str, Any]], tools: list[ToolView]) -> ParsedTurn: ...


class AdapterError(RuntimeError):
    """The back end failed or declined; no tool call should be inferred."""


@dataclass
class ScriptedAdapter:
    """Replays prepared turns; raises once the script is exhausted."""

    turns: list[ParsedTurn]
    name: str = "scripted"
    seen: list[tuple[list[dict[str, Any]], list[str]]] = field(default_factory=list)

    def next_turn(self, messages: list[dict[str, Any]], tools: list[ToolView]) -> ParsedTurn:
        self.seen.append((copy.deepcopy(messages), [tool.name for tool in tools]))
        if not self.turns:
            raise AdapterError("scripted adapter has no turns left")
        return self.turns.pop(0)


def split_tagged_reasoning(text: str) -> tuple[str, str]:
    """Teacher convention: an optional ``<reasoning>`` block, then the message."""
    match = _REASONING_TAG.search(text)
    if not match:
        return "", text.strip()
    reasoning = match.group(1).strip()
    remainder = (text[: match.start()] + text[match.end() :]).strip()
    return reasoning, remainder


class AnthropicAdapter:
    """Claude through ``anthropic.Anthropic().messages`` with client-side tools.

    The adapter keeps Claude's own content blocks (including thinking blocks) and
    only ever appends to that history, as preserved thinking requires. Internal
    messages are mirrored in order; a rewrite of earlier messages is refused.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        system_suffix: str = "",
        max_tokens: int = 16000,
        refusal_fallback: bool = True,
        rationale_field: str | None = None,
        client: Any = None,
    ):
        import anthropic

        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic(max_retries=4)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.refusal_fallback = refusal_fallback
        self.system_suffix = system_suffix
        # Opus 5.5 returns text written between tool calls as (hidden) thinking
        # blocks, while tool inputs always arrive verbatim. A teacher run therefore
        # asks for its per-call rationale as a required tool argument, removed
        # again before the call is validated.
        self.rationale_field = rationale_field
        self.name = f"anthropic:{model}"
        self.usage = Usage()
        self._history: list[dict[str, Any]] = []
        self._mirrored = 0

    def _append(self, role: str, blocks: list[dict[str, Any]]) -> None:
        if self._history and self._history[-1]["role"] == role:
            self._history[-1]["content"].extend(blocks)
        else:
            self._history.append({"role": role, "content": list(blocks)})

    def _mirror(self, messages: list[dict[str, Any]]) -> str:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        start = 1 if system else 0
        if self._mirrored < start:
            self._mirrored = start
        for message in messages[self._mirrored :]:
            role = message["role"]
            if role == "user":
                self._append("user", [{"type": "text", "text": message["content"]}])
            elif role == "tool":
                self._append(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message["content"],
                            **({"is_error": True} if message.get("is_error") else {}),
                        }
                    ],
                )
            elif role == "assistant":
                blocks = message.get("_provider_blocks")
                if blocks is None:
                    raise AdapterError("assistant turns must come from this adapter (append-only history)")
                self._append("assistant", blocks)
            else:
                raise AdapterError(f"unexpected role {role!r}")
        self._mirrored = len(messages)
        return system

    def _tool_definition(self, tool: ToolView) -> dict[str, Any]:
        definition = tool.as_anthropic()
        if not self.rationale_field:
            return definition
        schema = copy.deepcopy(definition["input_schema"])
        schema.setdefault("properties", {})[self.rationale_field] = {
            "type": "string",
            "description": "Private note, not shown to the user: one to three sentences on why this tool and these arguments, and what to check in the result.",
        }
        schema["required"] = [*schema.get("required", []), self.rationale_field]
        definition["input_schema"] = schema
        return definition

    def next_turn(self, messages: list[dict[str, Any]], tools: list[ToolView]) -> ParsedTurn:
        system = self._mirror(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system + self.system_suffix,
            "messages": self._history,
            "tools": [self._tool_definition(tool) for tool in tools],
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        try:
            if self.refusal_fallback:
                with self.client.beta.messages.stream(
                    **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                ) as stream:
                    response = stream.get_final_message()
            else:
                with self.client.messages.stream(**request) as stream:
                    response = stream.get_final_message()
        except self.anthropic.APIStatusError as error:
            raise AdapterError(f"Anthropic API error {error.status_code}: {error.message}") from error
        except self.anthropic.APIConnectionError as error:
            raise AdapterError("Anthropic API connection failed") from error

        usage = response.usage
        self.usage.add(
            Usage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                requests=1,
            )
        )
        if response.stop_reason == "refusal":
            raise AdapterError("model declined the request (refusal)")
        if response.stop_reason == "max_tokens":
            raise AdapterError("model output hit max_tokens; turn discarded")

        blocks = [block.to_dict() for block in response.content]
        text = "".join(block.text for block in response.content if block.type == "text")
        reasoning, content = split_tagged_reasoning(text)
        calls = []
        rationales = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            arguments = dict(block.input) if isinstance(block.input, dict) else block.input
            if self.rationale_field and isinstance(arguments, dict):
                note = arguments.pop(self.rationale_field, None)
                if isinstance(note, str) and note.strip():
                    rationales.append(note.strip())
            calls.append({"id": block.id, "name": block.name, "arguments": arguments})
        if rationales:
            reasoning = "\n".join(part for part in [reasoning, *rationales] if part)
        turn = ParsedTurn(reasoning=reasoning, content=content, tool_calls=calls)
        turn.provider_blocks = blocks  # type: ignore[attr-defined]
        turn.served_by = response.model  # type: ignore[attr-defined]
        return turn


class OpenAICompatibleAdapter:
    """Chat completions against an OpenAI-compatible server (no SDK dependency).

    Configure with ``STRUCTLLM_MODEL_URL`` (e.g. ``http://127.0.0.1:8000/v1``),
    ``STRUCTLLM_MODEL_NAME`` and optionally ``STRUCTLLM_MODEL_API_KEY``. The
    server must be started with its tool-call parser for the model, e.g. vLLM
    ``--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        tool_format: str = "qwen3_xml",
        enable_thinking: bool = True,
    ):
        import httpx2

        self.base_url = (base_url or os.environ.get("STRUCTLLM_MODEL_URL", "")).rstrip("/")
        self.model = model or os.environ.get("STRUCTLLM_MODEL_NAME", "")
        if not self.base_url or not self.model:
            raise ValueError("Set STRUCTLLM_MODEL_URL and STRUCTLLM_MODEL_NAME for the OpenAI-compatible adapter")
        key = api_key or os.environ.get("STRUCTLLM_MODEL_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx2.Client(headers=headers, timeout=timeout)
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.tool_format = tool_format
        self.enable_thinking = enable_thinking
        self.name = f"openai-compatible:{self.model}"
        self.usage = Usage()

    def next_turn(self, messages: list[dict[str, Any]], tools: list[ToolView]) -> ParsedTurn:
        payload_messages = []
        for message in to_template_messages(messages):
            item = dict(message)
            if item.get("tool_calls"):
                item["tool_calls"] = [
                    {**call, "function": {**call["function"], "arguments": json.dumps(call["function"]["arguments"])}}
                    for call in item["tool_calls"]
                ]
            payload_messages.append(item)
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "tools": [tool.as_function() for tool in tools],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        try:
            response = self.http.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as error:  # transport errors of any kind end the turn
            raise AdapterError(f"model endpoint failed: {type(error).__name__}") from error
        usage = data.get("usage") or {}
        self.usage.add(
            Usage(input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0), requests=1)
        )
        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        calls = []
        errors = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            try:
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append({"id": call.get("id") or f"call_{index + 1}", "name": function["name"], "arguments": arguments})
            except (KeyError, ValueError) as error:
                errors.append(f"tool call {index + 1}: {error}")
        if not calls and "<tool_call>" in content:
            # Server ran without a tool parser: parse the raw template output.
            parsed = parse_generation((f"<think>{reasoning}</think>" if reasoning else "") + content, self.tool_format)
            return parsed
        if not reasoning:
            parsed = parse_generation(content, self.tool_format)
            reasoning, content = parsed.reasoning, parsed.content
        return ParsedTurn(reasoning=reasoning.strip(), content=content.strip(), tool_calls=calls, parse_errors=errors)


class HuggingFaceAdapter:
    """Local generation with ``transformers`` and the tokenizer's chat template."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        tool_format: str,
        max_new_tokens: int = 2048,
        enable_thinking: bool = True,
        do_sample: bool = False,
        temperature: float = 0.6,
        top_p: float = 0.95,
        name: str = "huggingface",
    ):
        self.model, self.tokenizer = model, tokenizer
        self.tool_format = tool_format
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.do_sample, self.temperature, self.top_p = do_sample, temperature, top_p
        self.name = name
        self.usage = Usage()

    def render_prompt(self, messages: list[dict[str, Any]], tools: list[ToolView]) -> str:
        return self.tokenizer.apply_chat_template(
            to_template_messages(messages),
            tools=[tool.as_function() for tool in tools] or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate_text(self, prompt: str) -> tuple[str, int, int]:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        options: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            options.update(temperature=self.temperature, top_p=self.top_p)
        with torch.inference_mode():
            output = self.model.generate(**encoded, **options, pad_token_id=self.tokenizer.pad_token_id)
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        for token in (self.tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"):
            if token:
                text = text.split(token)[0]
        return text, int(encoded["input_ids"].shape[1]), int(new_tokens.shape[0])

    def next_turn(self, messages: list[dict[str, Any]], tools: list[ToolView]) -> ParsedTurn:
        prompt = self.render_prompt(messages, tools)
        # The generation prompt may already open the reasoning block.
        prefix = "<think>\n" if prompt.rstrip().endswith("<think>") else ""
        text, prompt_tokens, new_tokens = self.generate_text(prompt)
        self.usage.add(Usage(input_tokens=prompt_tokens, output_tokens=new_tokens, requests=1))
        return parse_generation(prefix + text, self.tool_format)
