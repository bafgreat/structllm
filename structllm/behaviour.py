"""Score a generated assistant turn against the teacher's turn.

Eval loss says the tokens were learned. It does not say the model picks the right
tool, writes arguments the schema accepts, copies the revision id from earlier in
the conversation, or states only numbers the tools returned. These are scored
from the generated text alone, so they can run inside the training loop on a
GPU node with no StructSub server.

A turn is scored against the teacher's turn for the same prompt:

* ``decision``    called a tool when the teacher did, answered when it did not
* ``names``       the same tools in the same order
* ``args_valid``  every argument object passes the tool's JSON schema
* ``args_exact``  the same arguments as the teacher (ids included, idempotency keys ignored)
* ``grounded``    no number in the reply lacks a source in the conversation
* ``known_tools`` every tool it named was offered
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from structllm.chat_format import ParsedTurn
from structllm.datagen.grounding import ungrounded

IGNORED_ARGUMENTS = frozenset({"idempotency_key"})


def _function(call: dict[str, Any]) -> dict[str, Any]:
    return call.get("function", call)


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = _function(call).get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def expected_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The teacher's tool calls in an exported assistant message."""
    return [{"name": _function(call).get("name"), "arguments": _arguments(call)} for call in message.get("tool_calls") or []]


def schemas_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parameter schema for each offered tool, from function definitions in either common shape."""
    out = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        name = function.get("name")
        if name:
            out[name] = function.get("parameters") or function.get("input_schema") or function.get("inputSchema") or {}
    return out


def _normalise(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return {key: _normalise(value) for key, value in sorted(arguments.items()) if key not in IGNORED_ARGUMENTS}
    if isinstance(arguments, list):
        return [_normalise(value) for value in arguments]
    return arguments


def _valid(schema: dict[str, Any], arguments: dict[str, Any]) -> bool:
    if not schema:
        return True
    try:
        return Draft202012Validator(schema).is_valid(arguments)
    except SchemaError:
        return True  # a malformed offered schema is not the model's fault


def score_turn(
    parsed: ParsedTurn,
    expected: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prefix: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scores for one generated turn. ``None`` means the measure does not apply to this turn."""
    schemas = schemas_by_name(tools)
    got = [{"name": call.get("name"), "arguments": call.get("arguments") or {}} for call in parsed.tool_calls]
    score: dict[str, Any] = {
        "parsed": not parsed.parse_errors,
        "decision": bool(got) == bool(expected),
        "names": None,
        "args_valid": None,
        "args_exact": None,
        "known_tools": None,
        "grounded": None,
    }
    if expected:
        names_match = [c["name"] for c in got] == [c["name"] for c in expected]
        score["names"] = names_match
        score["args_valid"] = bool(got) and all(_valid(schemas.get(c["name"], {}), c["arguments"]) for c in got)
        score["args_exact"] = names_match and [_normalise(c["arguments"]) for c in got] == [_normalise(c["arguments"]) for c in expected]
    if got:
        score["known_tools"] = all(c["name"] in schemas for c in got)
    if prefix is not None and not got and parsed.content.strip():
        reply = {"role": "assistant", "content": parsed.content}
        score["grounded"] = not any(index == len(prefix) for index, _ in ungrounded([*prefix, reply], tools))
    return score


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates over the turns each measure applies to, plus one number for choosing a checkpoint."""
    rates: dict[str, Any] = {"turns": len(scores)}
    for key in ("parsed", "decision", "names", "args_valid", "args_exact", "known_tools", "grounded"):
        applicable = [s[key] for s in scores if s.get(key) is not None]
        rates[key] = round(sum(applicable) / len(applicable), 4) if applicable else None
    parts = [rates[k] for k in ("decision", "names", "args_valid", "args_exact") if rates[k] is not None]
    rates["score"] = round(sum(parts) / len(parts), 4) if parts else 0.0
    return rates
