"""Grounding: every number the assistant states must trace back to this conversation.

StructSub's rule is that the model never produces a scientific number, it relays
them. A teacher that writes "porosity 0.41" when no tool said so would otherwise be
accepted and trained on. This checks assistant text against what the conversation
supplied: the user's own messages, the arguments the assistant sent, and tool results.

A number is grounded when some supplied number matches it to the precision it is
written at, either as given or under a plain rescaling (fraction to percent,
J to kJ, angstrom to nm). Small whole numbers (at most ``SMALL_INTEGER``) are
treated as counting words and list numbering and are not checked, and numbers that
are part of an identifier (a SMILES string, a UUID, a formula such as Zr6) are
skipped because they sit inside a word.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

SMALL_INTEGER = 10
# as given, fraction <-> percent, J <-> kJ, angstrom <-> nm
SCALES = (1.0, 100.0, 0.01, 1000.0, 0.001, 10.0, 0.1)

# Not preceded or followed by a word character or hyphen, so identifiers do not count.
_NUMBER = re.compile(r"(?<![\w.\-+])[-−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w\-]|\.\d)")
_ANY_NUMBER = re.compile(r"[-−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse(token: str) -> tuple[float, int]:
    token = token.replace("−", "-").replace(",", "")
    mantissa = token.lower().split("e")[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return float(token), decimals


def claimed_numbers(text: str) -> list[tuple[str, float, int]]:
    """Numbers an assistant message states, with the decimals they are written to."""
    found = []
    for match in _NUMBER.finditer(text):
        token = match.group(0)
        try:
            value, decimals = _parse(token)
        except ValueError:
            continue
        found.append((token, value, decimals))
    return found


def _leaves(value: Any, out: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            out.append(float(value))
    elif isinstance(value, str):
        for match in _ANY_NUMBER.finditer(value):
            try:
                out.append(float(match.group(0).replace("−", "-")))
            except ValueError:
                pass
    elif isinstance(value, dict):
        for item in value.values():
            _leaves(item, out)
    elif isinstance(value, list):
        for item in value:
            _leaves(item, out)


def supplied_numbers(messages: list[dict[str, Any]]) -> list[float]:
    """Every number carried by user messages, assistant tool arguments and tool results."""
    out: list[float] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            _leaves(message.get("content") or "", out)
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                _leaves(call.get("arguments"), out)
        elif role == "tool":
            content = message.get("content")
            try:
                _leaves(json.loads(content) if isinstance(content, str) else content, out)
            except (TypeError, ValueError):
                _leaves(content, out)
    return out


def is_grounded(value: float, decimals: int, supplied: list[float]) -> bool:
    tolerance = 0.5 * 10 ** (-decimals)
    for source in supplied:
        # the mantissa too, for values written as "1.741 x 10^-4" from 0.0001741
        mantissa = source / 10.0 ** math.floor(math.log10(abs(source))) if 1e-290 < abs(source) < 1e290 else 0.0
        for scale in SCALES:
            for candidate in (source * scale, mantissa * scale):
                if abs(candidate - value) <= tolerance * (1 + 1e-9) + 1e-12 * abs(candidate):
                    return True
    return False


def _pair_derived(numbers: list[float]) -> list[float]:
    """Differences, sums and ratios of two supplied numbers: arithmetic an answer may legitimately do."""
    unique = sorted({round(n, 9) for n in numbers})
    if len(unique) > 160:
        return []
    out = []
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            out.extend((b - a, a + b))
            if a:
                out.append(b / a)
            if b:
                out.append(a / b)
    return out


def ungrounded(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[tuple[int, list[str]]]:
    """(message index, numbers) for each assistant message that states unsupported numbers.

    ``tools`` are the function definitions offered; defaults and limits quoted from them
    are supplied by the platform. A message in a turn where no tool ran can only offer
    examples or repeat earlier results, so it is checked against earlier results only
    and example values the user never gave are not held against it.
    """
    tool_numbers: list[float] = []
    if tools:
        _leaves(tools, tool_numbers)
    problems = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        text = message.get("content") or ""
        if not text.strip():
            continue
        last_user = max((i for i in range(index) if messages[i].get("role") == "user"), default=-1)
        ran_tool = any(m.get("role") == "tool" for m in messages[last_user + 1:index])
        if not ran_tool:
            continue
        supplied = supplied_numbers(messages[:index]) + tool_numbers
        supplied += _pair_derived(supplied_numbers(messages[last_user + 1:index]))
        bad = []
        for token, value, decimals in claimed_numbers(text):
            if decimals == 0 and abs(value) <= SMALL_INTEGER:
                continue
            if not is_grounded(value, decimals, supplied):
                bad.append(token)
        if bad:
            problems.append((index, bad))
    return problems
