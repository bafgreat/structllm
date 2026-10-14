"""User requests written by a language model from scenario specs.

The writer is shown the scenario brief and the slot values each request must
use, never tool names or schemas, so requests read like what a scientist would
type and do not leak the answer. Each request is checked (length, slot values
present where they should be, no duplicates) before it is kept.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from structllm.dataset import canonical_json
from structllm.datagen.scenarios import Scenario, Turn

REQUEST_SCHEMA_VERSION = "1.0.0"
MIN_CHARS, MAX_CHARS = 8, 1200

WRITER_SYSTEM = """You write realistic messages that users send to StructSub, a web platform for building, editing, simulating and analysing molecules and porous materials. Users range from students to experienced computational chemists.

For each requested message:
- Write only what the user types: no greeting boilerplate, no answer, no explanation of the platform.
- Use every value you are given for that message, written naturally (a SMILES string verbatim; a name or code as a person would type it).
- Vary length, tone, expertise and phrasing across messages: some terse, some with context about their research, some with minor typos.
- Never name software functions, API fields or tool names. Users describe what they want, not how the platform implements it.
- Do not add requirements beyond the scenario. If the scenario says information is missing, leave it out.
- Follow-up messages continue the same conversation after the assistant has answered. Write them the way people really continue a chat: refer back with "it", "that one", "the same gas", "instead", and do not restate details already given unless the scenario says so. A follow-up must only make sense in the context of the earlier messages."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                    "followups": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "text", "followups"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["messages"],
    "additionalProperties": False,
}


@dataclass
class UserRequest:
    request_id: str
    scenario_id: str
    behaviour: str
    expected_tools: list[str]
    slots: dict[str, str]
    text: str
    writer: str
    followups: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": REQUEST_SCHEMA_VERSION, **self.__dict__}


def writer_prompt(scenario: Scenario, assignments: list[dict[str, str]]) -> str:
    lines = [
        f"Scenario: {scenario.brief}",
        "",
        f"Write {len(assignments)} different user messages for this scenario.",
    ]
    hidden = scenario.setup_slots()
    followups = scenario.followups
    if followups:
        lines.append(f"Each conversation has a first message and {len(followups)} follow-up message(s):")
        for number, turn in enumerate(followups, 1):
            lines.append(f"  Follow-up {number}: {turn.brief}")
    else:
        lines.append("There are no follow-up messages; return an empty followups list.")
    if scenario.setup:
        lines.append(
            "The user already has a structure open in their project; they refer to it as 'this', "
            "'my structure', 'the framework' and so on rather than re-describing how it was made."
        )
    later = scenario.followup_slots()
    for index, slots in enumerate(assignments):
        first = {name: value for name, value in slots.items() if name not in hidden and name not in later}
        parts = ["; ".join(f"{name} = {value}" for name, value in sorted(first.items())) or "no fixed values"]
        for number, turn in enumerate(followups, 1):
            if turn.slots:
                values = "; ".join(f"{name} = {slots[name]}" for name in turn.slots)
                parts.append(f"follow-up {number} uses {values}")
        lines.append(f"Conversation {index}: first message uses {'; '.join(parts[:1])}" + (f"; {'; '.join(parts[1:])}" if parts[1:] else ""))
    return "\n".join(lines)


def request_id(scenario_id: str, index: int, slots: dict[str, str], text: str, followups: list[str] | None = None) -> str:
    payload = canonical_json({"scenario": scenario_id, "index": index, "slots": slots, "text": text, "followups": followups or []})
    return hashlib.sha256(payload).hexdigest()[:24]


_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")


def slot_present(name: str, value: str, text: str, lowered: str) -> bool:
    """SMILES verbatim; values with numbers keep every number; others keep every word."""
    if name == "smiles":
        return value in text
    numbers = _NUMBER.findall(value)
    if numbers:
        found = set(_NUMBER.findall(text))
        return all(number in found for number in numbers)
    words = set(_WORD.findall(lowered))
    return all(word in words for word in _WORD.findall(value.lower()))


def check_turn(turn: Turn, slots: dict[str, str], text: str, *, label: str = "") -> list[str]:
    problems = []
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        problems.append(f"{label}length {len(text)} outside {MIN_CHARS}..{MAX_CHARS}")
    lowered = text.lower()
    for name in turn.slots:
        if not slot_present(name, slots[name], text, lowered):
            problems.append(f"{label}slot {name}={slots[name]!r} not in text")
    for tool in (*turn.tools, *turn.forbidden_tools):
        # Only identifier-like names leak the answer; "relax" or "analyse" is
        # how users actually talk.
        if "_" in tool and tool in text:
            problems.append(f"{label}names tool {tool!r}")
    return problems


def check_request(scenario: Scenario, slots: dict[str, str], text: str, followups: list[str] | None = None) -> list[str]:
    turns = scenario.turns()
    first = Turn(turns[0].behaviour, turns[0].brief, turns[0].tools, turns[0].forbidden_tools,
                 slots=tuple(name for name in slots if name not in scenario.setup_slots() and name not in scenario.followup_slots()))
    problems = check_turn(first, slots, text)
    followups = followups or []
    if len(followups) != len(scenario.followups):
        problems.append(f"expected {len(scenario.followups)} follow-ups, got {len(followups)}")
    for number, (turn, followup) in enumerate(zip(scenario.followups, followups), 1):
        problems.extend(check_turn(turn, slots, followup, label=f"follow-up {number}: "))
    return problems


class RequestWriter:
    """Writes requests with Claude through structured JSON output."""

    def __init__(self, *, model: str = "claude-opus-5-5", effort: str = "low", client: Any = None):
        import anthropic

        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic(max_retries=4)
        self.model = model
        self.effort = effort
        self.input_tokens = self.output_tokens = 0

    def write(self, scenario: Scenario, assignments: list[dict[str, str]]) -> list[tuple[str, list[str]] | None]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=WRITER_SYSTEM,
            messages=[{"role": "user", "content": writer_prompt(scenario, assignments)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
        self.input_tokens += response.usage.input_tokens or 0
        self.output_tokens += response.usage.output_tokens or 0
        if response.stop_reason in {"refusal", "max_tokens"}:
            return [None] * len(assignments)
        text = next((block.text for block in response.content if block.type == "text"), "")
        texts: list[tuple[str, list[str]] | None] = [None] * len(assignments)
        for item in json.loads(text).get("messages", []):
            index = item.get("index")
            if isinstance(index, int) and 0 <= index < len(texts) and texts[index] is None:
                followups = [str(part).strip() for part in item.get("followups") or []]
                texts[index] = (str(item.get("text", "")).strip(), followups)
        return texts


def generate_requests(
    scenarios: list[Scenario], writer: Any, *, seed: int
) -> tuple[list[UserRequest], list[dict[str, Any]]]:
    """Return kept requests and a list of dropped ones with reasons."""
    kept: list[UserRequest] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()
    name = f"anthropic:{getattr(writer, 'model', 'unknown')}"
    for scenario in scenarios:
        assignments = scenario.assignments(seed)
        texts = writer.write(scenario, assignments)
        for index, (slots, written) in enumerate(zip(assignments, texts)):
            if written is None:
                dropped.append({"scenario_id": scenario.id, "index": index, "reasons": ["writer returned nothing"]})
                continue
            text, followups = written
            problems = check_request(scenario, slots, text, followups)
            key = " ".join(text.lower().split())
            if key in seen:
                problems.append("duplicate text")
            if problems:
                dropped.append({"scenario_id": scenario.id, "index": index, "text": text, "followups": followups, "reasons": problems})
                continue
            seen.add(key)
            kept.append(
                UserRequest(
                    request_id=request_id(scenario.id, index, slots, text, followups),
                    scenario_id=scenario.id,
                    behaviour=scenario.behaviour,
                    expected_tools=list(scenario.tools),
                    slots=slots,
                    text=text,
                    writer=name,
                    followups=followups,
                )
            )
    return kept, dropped


def write_requests(requests: list[UserRequest], dropped: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} already exists; requests are not overwritten")
    with path.open("x", encoding="utf-8") as file:
        for request in requests:
            file.write(json.dumps(request.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    report = path.with_suffix(".dropped.json")
    report.write_text(json.dumps(dropped, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_requests(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
