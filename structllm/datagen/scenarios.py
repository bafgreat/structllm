"""Scenario specs: what a synthetic user wants, and what a good answer does.

Specs live in ``structllm/resources/scenarios.yaml``. Loading checks every
named tool against the catalogue the data will be generated from, so a renamed
or removed capability fails loudly instead of producing silently wrong data.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from structllm.toolspec import Catalogue

BEHAVIOURS = frozenset({"execute", "clarify", "approval_hold", "unsupported", "failure", "discuss"})
MAX_VARIANTS = 50


class ScenarioError(ValueError):
    """A scenario file is malformed or does not fit the catalogue."""


@dataclass(frozen=True)
class SetupStep:
    tool: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Turn:
    """One user message and what a good answer to it does.

    ``forbidden_tools`` catches answers driven by words rather than meaning
    ("don't relax it, just export it" must not relax). ``uses_prior_revision``
    requires structure arguments to name a revision that already appeared in
    the conversation, so "it" has to be resolved from context.
    """

    behaviour: str
    brief: str
    tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    uses_prior_revision: bool = False
    slots: tuple[str, ...] = ()  # slot names this follow-up must mention


@dataclass(frozen=True)
class Scenario:
    id: str
    behaviour: str
    brief: str
    tools: tuple[str, ...] = ()
    setup: tuple[SetupStep, ...] = ()
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    variants: int = 3
    forbidden_tools: tuple[str, ...] = ()
    uses_prior_revision: bool = False
    followups: tuple[Turn, ...] = ()
    zip_slots: bool = False  # slot lists are parallel (value i of each belongs together)

    def turns(self) -> list[Turn]:
        first_slots = tuple(name for name in self.slots if name not in self.followup_slots())
        first = Turn(self.behaviour, self.brief, self.tools, self.forbidden_tools, self.uses_prior_revision, first_slots)
        return [first, *self.followups]

    def followup_slots(self) -> set[str]:
        return {name for turn in self.followups for name in turn.slots}

    def setup_slots(self) -> set[str]:
        """Slots used only to prepare the starting structure."""
        used = set()
        for step in self.setup:
            text = repr(step.parameters)
            used.update(name for name in self.slots if "{" + name + "}" in text)
        return used

    def assignments(self, seed: int) -> list[dict[str, str]]:
        """One slot assignment per variant, cycling through shuffled values.

        Deterministic for a seed, so a request file can be regenerated exactly.
        """
        rng = random.Random(int(hashlib.sha256(f"{seed}:{self.id}".encode()).hexdigest()[:16], 16))
        if self.zip_slots and self.slots:
            size = len(next(iter(self.slots.values())))
            permutation = list(range(size))
            rng.shuffle(permutation)
            return [
                {name: values[permutation[index % size]] for name, values in self.slots.items()}
                for index in range(self.variants)
            ]
        orders = {}
        for name, values in sorted(self.slots.items()):
            order = list(values)
            rng.shuffle(order)
            orders[name] = order
        return [{name: order[index % len(order)] for name, order in orders.items()} for index in range(self.variants)]


def fill(value: Any, slots: dict[str, str]) -> Any:
    """Substitute ``{slot}`` placeholders in strings inside a parameter object."""
    if isinstance(value, str):
        for name, slot_value in slots.items():
            value = value.replace("{" + name + "}", slot_value)
        return value
    if isinstance(value, dict):
        return {key: fill(item, slots) for key, item in value.items()}
    if isinstance(value, list):
        return [fill(item, slots) for item in value]
    return value


def _names(item: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = item.get(key) or []
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ScenarioError(f"{where}: {key!r} must be a list of names")
    return tuple(value)


def _behaviour(item: dict[str, Any], where: str) -> tuple[str, str, tuple[str, ...]]:
    behaviour = item.get("behaviour")
    if behaviour not in BEHAVIOURS:
        raise ScenarioError(f"{where}: behaviour must be one of {sorted(BEHAVIOURS)}")
    brief = item.get("brief")
    if not isinstance(brief, str) or not brief.strip():
        raise ScenarioError(f"{where}: 'brief' is required")
    tools = _names(item, "tools", where)
    if behaviour in {"execute", "approval_hold", "failure"} and not tools:
        raise ScenarioError(f"{where}: behaviour {behaviour!r} needs at least one expected tool")
    return behaviour, brief.strip(), tools


def _parse(item: Any, index: int) -> Scenario:
    if not isinstance(item, dict):
        raise ScenarioError(f"scenario {index} is not a mapping")
    scenario_id = item.get("id")
    if not isinstance(scenario_id, str) or not scenario_id.isidentifier():
        raise ScenarioError(f"scenario {index}: 'id' must be an identifier")
    behaviour, brief, tools = _behaviour(item, scenario_id)
    setup = []
    for step in item.get("setup") or []:
        if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
            raise ScenarioError(f"{scenario_id}: each setup step needs a 'tool'")
        setup.append(SetupStep(step["tool"], dict(step.get("parameters") or {})))
    slots = {}
    for name, values in (item.get("slots") or {}).items():
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            raise ScenarioError(f"{scenario_id}: slot {name!r} needs a non-empty list of strings")
        slots[str(name)] = tuple(values)
    variants = item.get("variants", 3)
    if not isinstance(variants, int) or not 1 <= variants <= MAX_VARIANTS:
        raise ScenarioError(f"{scenario_id}: variants must be 1..{MAX_VARIANTS}")
    followups = []
    for number, turn in enumerate(item.get("followups") or [], 1):
        where = f"{scenario_id} follow-up {number}"
        if not isinstance(turn, dict):
            raise ScenarioError(f"{where} is not a mapping")
        turn_behaviour, turn_brief, turn_tools = _behaviour(turn, where)
        turn_slots = _names(turn, "slots", where)
        unknown = set(turn_slots) - set(slots)
        if unknown:
            raise ScenarioError(f"{where}: unknown slots {sorted(unknown)}")
        followups.append(Turn(turn_behaviour, turn_brief, turn_tools, _names(turn, "forbidden_tools", where),
                              bool(turn.get("uses_prior_revision")), turn_slots))
    zip_slots = bool(item.get("zip_slots"))
    if zip_slots and len({len(values) for values in slots.values()}) > 1:
        raise ScenarioError(f"{scenario_id}: zip_slots needs slot lists of equal length")
    return Scenario(
        scenario_id, behaviour, brief, tools, tuple(setup), slots, variants,
        forbidden_tools=_names(item, "forbidden_tools", scenario_id),
        uses_prior_revision=bool(item.get("uses_prior_revision")),
        followups=tuple(followups),
        zip_slots=zip_slots,
    )


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    if path is None:
        text = resources.files("structllm").joinpath("resources/scenarios.yaml").read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("scenarios"), list):
        raise ScenarioError("scenario file must have version: 1 and a 'scenarios' list")
    scenarios = [_parse(item, index) for index, item in enumerate(data["scenarios"])]
    seen = set()
    for scenario in scenarios:
        if scenario.id in seen:
            raise ScenarioError(f"duplicate scenario id {scenario.id!r}")
        seen.add(scenario.id)
    return scenarios


def check_against_catalogue(scenarios: list[Scenario], catalogue: Catalogue) -> None:
    """Every expected and setup tool must exist; setup parameters must be schema-valid."""
    from structllm.toolspec import ToolCallError, prepare_call

    placeholder_project = "00000000-0000-4000-8000-000000000000"
    problems = []
    for scenario in scenarios:
        for turn in scenario.turns():
            for name in (*turn.tools, *turn.forbidden_tools):
                if name not in catalogue:
                    problems.append(f"{scenario.id}: tool {name!r} is not in the catalogue")
        for slots in scenario.assignments(seed=0):
            for step in scenario.setup:
                if step.tool not in catalogue:
                    problems.append(f"{scenario.id}: setup tool {step.tool!r} is not in the catalogue")
                    continue
                arguments = {"project_id": placeholder_project, "parameters": fill(step.parameters, slots)}
                if "structure_revision_id" in catalogue.get(step.tool).get("inputSchema", {}).get("properties", {}):
                    arguments["structure_revision_id"] = placeholder_project  # setup passes the previous revision at run time
                try:
                    prepare_call(catalogue, name=step.tool, arguments=arguments, allowed={step.tool}, session_id="check", step=0)
                except ToolCallError as error:
                    problems.append(f"{scenario.id}: setup {step.tool} is invalid: {error} {error.details[:2]}")
    if problems:
        raise ScenarioError("; ".join(sorted(set(problems))))
