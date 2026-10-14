"""Provider-neutral conversations and native chat-template tool calls.

Internal message shapes (what datasets store and the orchestrator produces)::

    {"role": "system" | "user", "content": str}
    {"role": "assistant", "content": str, "reasoning": str | None,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}], "train": bool}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

``train: false`` marks an assistant turn kept for context but excluded from the
loss (for example a teacher's call that the orchestrator rejected).

Generated text is parsed with the format of the model's own chat template:
``qwen3_xml`` (Qwen3.5 / Qwen3-Coder: ``<function=...><parameter=...>``) or
``hermes`` (Qwen3 / Qwen2.5: JSON inside ``<tool_call>``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)
_XML_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|\Z)", re.DOTALL)
_XML_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


@dataclass
class ParsedTurn:
    """One generated assistant turn, split into its parts."""

    reasoning: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate ``<think>`` reasoning from the visible answer.

    The generation prompt of thinking templates already contains ``<think>\\n``,
    so generated text may start inside the reasoning and contain only the close tag.
    """
    if THINK_CLOSE in text:
        head, tail = text.split(THINK_CLOSE, 1)
        reasoning = head.split(THINK_OPEN, 1)[-1]
        return reasoning.strip(), tail.strip()
    if text.lstrip().startswith(THINK_OPEN):
        # Reasoning never closed (cut off by the token limit): there is no answer.
        return text.split(THINK_OPEN, 1)[1].strip(), ""
    return "", text.strip()


def _parse_xml_call(block: str) -> dict[str, Any]:
    match = _XML_FUNCTION.search(block)
    if not match:
        raise ValueError("tool_call block has no <function=...> element")
    name = match.group(1).strip()
    arguments: dict[str, Any] = {}
    for key, raw in _XML_PARAMETER.findall(match.group(2)):
        key = key.strip()
        if key in arguments:
            raise ValueError(f"parameter {key!r} given twice")
        value = raw.strip("\n")
        # Objects and arrays are rendered as JSON by the template; scalars as text.
        # Leave scalar text alone here: the live schema decides how to read it.
        if value[:1] in "[{":
            try:
                arguments[key] = json.loads(value)
                continue
            except ValueError:
                pass
        arguments[key] = value
    return {"name": name, "arguments": arguments}


def _parse_hermes_call(block: str) -> dict[str, Any]:
    payload = json.loads(block.strip())
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise ValueError("tool_call JSON must be an object with a name")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return {"name": payload["name"], "arguments": arguments}


def parse_generation(text: str, tool_format: str) -> ParsedTurn:
    """Parse raw generated text (special tokens already removed)."""
    reasoning, visible = split_reasoning(text)
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    first = visible.find("<tool_call>")
    content = visible if first < 0 else visible[:first]
    for index, block in enumerate(_TOOL_BLOCK.findall(visible)):
        try:
            call = _parse_xml_call(block) if tool_format == "qwen3_xml" else _parse_hermes_call(block)
        except (ValueError, TypeError) as error:
            errors.append(f"tool call {index + 1}: {error}")
            continue
        call["id"] = f"call_{index + 1}"
        calls.append(call)
    return ParsedTurn(reasoning=reasoning, content=content.strip(), tool_calls=calls, parse_errors=errors)


def to_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Internal messages -> the message list Hugging Face chat templates consume."""
    converted = []
    for message in messages:
        role = message["role"]
        if role in {"system", "user"}:
            converted.append({"role": role, "content": message["content"]})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("reasoning"):
                item["reasoning_content"] = message["reasoning"]
            if message.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "id": call.get("id"),
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in message["tool_calls"]
                ]
            converted.append(item)
        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "content": message["content"],
                    "tool_call_id": message.get("tool_call_id"),
                    "name": message.get("name"),
                }
            )
        else:
            raise ValueError(f"Unsupported message role {role!r}")
    return converted


def tool_result_text(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
