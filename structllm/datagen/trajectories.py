"""Teacher trajectories: a model answers each request through the orchestrator.

For every request a fresh development project is created, the scenario's cheap
setup calls are run, and the teacher answers through the same ``Session`` the
trained model will use, under the generation policy (expensive work gets a
mock approval hold and is never sent). The result is one training record:
the transcript, the exact tool definitions offered, and for every tool result
whether it was executed or mocked.

Records are checked structurally against the scenario's expected behaviour.
The checks say whether the conversation has the right shape (asked instead of
guessing, reported a hold instead of polling it, used an expected tool); they
do not establish that any scientific statement in it is correct.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from structllm.adapters import ModelAdapter
from structllm.dataset import canonical_json
from structllm.datagen.policy import GenerationLimits, generation_policy, ml_engines_from_listing, needs_simulated_approval
from structllm.datagen.scenarios import Scenario, Turn, fill
from structllm.mcp_client import ToolTransport, sanitize
from structllm.orchestrator import IN_PROGRESS, Limits, Session, TurnRecord
from structllm.toolspec import Catalogue, ToolCallError, capability_meta, prepare_call, tool_version

TRAJECTORY_SCHEMA_VERSION = "1.0.0"
FAILED_OUTCOMES = frozenset({"model_error", "turn_limit", "time_limit", "unparseable_output", "incomplete"})


class SetupError(RuntimeError):
    """The starting state for a request could not be prepared; the teacher was not run."""


@dataclass
class SetupLimits:
    max_polls: int = 20
    poll_interval_seconds: float = 3.0


def scenario_split(scenario_id: str, *, seed: int, eval_fraction: float) -> str:
    """Whole scenarios go to one split, so evaluation requests are not paraphrases of training ones."""
    value = int(hashlib.sha256(f"{seed}:{scenario_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if value < eval_fraction else "train"


async def _call_and_wait(
    transport: ToolTransport, catalogue: Catalogue, name: str, arguments: dict[str, Any], limits: SetupLimits, sleep: Callable[[float], Any]
) -> dict[str, Any]:
    result = await transport.call_tool(name, arguments)
    for _ in range(limits.max_polls):
        if not (isinstance(result, dict) and result.get("status") in IN_PROGRESS and isinstance(result.get("job_id"), str)):
            break
        if "calculation_get" not in catalogue:
            break
        await sleep(limits.poll_interval_seconds)
        result = await transport.call_tool("calculation_get", {"job_id": result["job_id"]})
    return result if isinstance(result, dict) else {"status": "unknown"}


async def create_project(transport: ToolTransport, catalogue: Catalogue, name: str) -> str:
    arguments = prepare_call(catalogue, name="project_create", arguments={"name": name[:255]}, allowed={"project_create"}, session_id="setup", step=0)
    result = await transport.call_tool("project_create", arguments)
    project_id = result.get("project_id") if isinstance(result, dict) else None
    if not isinstance(project_id, str):
        raise SetupError("project_create returned no project_id")
    return project_id


async def run_setup(
    transport: ToolTransport,
    catalogue: Catalogue,
    scenario: Scenario,
    slots: dict[str, str],
    *,
    project_id: str,
    request_id: str,
    generation: GenerationLimits,
    limits: SetupLimits | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Run setup steps in order; return the last structure revision and a log."""
    limits = limits or SetupLimits()
    revision: str | None = None
    log = []
    for index, step in enumerate(scenario.setup):
        arguments: dict[str, Any] = {"project_id": project_id, "parameters": fill(step.parameters, slots)}
        properties = catalogue.get(step.tool).get("inputSchema", {}).get("properties", {}) if step.tool in catalogue else {}
        if "structure_revision_id" in properties:
            arguments["structure_revision_id"] = revision
        try:
            sent = prepare_call(catalogue, name=step.tool, arguments=arguments, allowed={step.tool}, session_id=f"setup-{request_id}", step=index)
        except ToolCallError as error:
            raise SetupError(f"setup {step.tool}: {error}") from error
        if needs_simulated_approval(step.tool, sent, generation):
            # Setup is part of the starting state, so it must really run; an
            # expensive setup step is a scenario bug, not something to mock.
            raise SetupError(f"setup {step.tool} would need approval; setup must be cheap")
        result = await _call_and_wait(transport, catalogue, step.tool, sent, limits, sleep)
        status = result.get("status")
        log.append({"tool": step.tool, "status": status, "tool_version": tool_version(catalogue.get(step.tool))})
        if status != "succeeded":
            raise SetupError(f"setup {step.tool} ended with status {status!r}")
        if isinstance(result.get("structure_revision_id"), str):
            revision = result["structure_revision_id"]
    return revision, log


async def discover_ml_engines(transport: ToolTransport, catalogue: Catalogue, project_id: str) -> frozenset[str]:
    """Engine names the server itself reports as ML potentials (read-only call)."""
    if "list_engines" not in catalogue:
        return frozenset()
    arguments = prepare_call(catalogue, name="list_engines", arguments={"project_id": project_id, "parameters": {}}, allowed={"list_engines"}, session_id="engines", step=0)
    result = await _call_and_wait(transport, catalogue, "list_engines", arguments, SetupLimits(), asyncio.sleep)
    properties = result.get("properties") if isinstance(result.get("properties"), dict) else {}
    return ml_engines_from_listing(properties)


# ----- checks ----------------------------------------------------------

def _effect(catalogue: Catalogue, name: str | None) -> str | None:
    if not name or name not in catalogue:
        return None
    return capability_meta(catalogue.get(name)).get("effect")


def _revisions_in(value: Any) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"structure_revision_id", "original_revision_id"} and isinstance(item, str):
                found.add(item)
            else:
                found |= _revisions_in(item)
    elif isinstance(value, list):
        for item in value:
            found |= _revisions_in(item)
    return found


def check_turn(turn: Turn, record: TurnRecord, catalogue: Catalogue, known_revisions: set[str]) -> list[str]:
    """Structural problems with one answered user message; empty means accepted.

    ``known_revisions`` holds the structure revisions the conversation has seen
    so far (from setup and earlier results); it is updated in place.
    """
    problems = []
    if record.outcome in FAILED_OUTCOMES:
        problems.append(f"outcome {record.outcome}")
    if not record.final_message.strip():
        problems.append("no final message")
    if any(call.decision == "denied" for call in record.calls):
        problems.append("called a denied tool")
    expected, forbidden = set(turn.tools), set(turn.forbidden_tools)
    for call in record.calls:
        if call.decision not in {"executed", "simulated"}:
            continue
        if call.live_tool in forbidden:
            problems.append(f"called {call.live_tool}, which this request rules out")
        target = (call.sent_arguments or {}).get("structure_revision_id")
        if turn.uses_prior_revision and call.live_tool in expected and target not in known_revisions:
            problems.append(f"{call.live_tool} acted on a revision not from this conversation")
        if call.executed:
            known_revisions |= _revisions_in(call.result)
    scientific = [call for call in record.calls if call.decision in {"executed", "simulated"}]
    writes = [call for call in scientific if _effect(catalogue, call.live_tool) != "read"]
    behaviour = turn.behaviour
    if behaviour == "execute":
        if not any(call.live_tool in expected and call.status == "succeeded" for call in scientific):
            problems.append("no expected tool succeeded")
        if record.outcome not in {"completed", "job_pending"}:
            problems.append(f"expected completion, got {record.outcome}")
    elif behaviour == "approval_hold":
        if record.outcome != "awaiting_approval":
            problems.append(f"expected awaiting_approval, got {record.outcome}")
        if not any(call.live_tool in expected and call.status == "awaiting_approval" for call in scientific):
            problems.append("expected tool was not held for approval")
    elif behaviour == "clarify":
        if writes:
            problems.append("ran operations instead of asking")
        if "?" not in record.final_message:
            problems.append("final message asks no question")
    elif behaviour == "unsupported":
        if writes:
            problems.append("ran operations for an unsupported request")
    elif behaviour == "discuss":
        if writes:
            problems.append("ran operations instead of answering from the conversation")
    elif behaviour == "failure":
        if not any(call.live_tool in expected and call.status == "failed" and call.executed for call in scientific):
            problems.append("expected tool did not fail")
    return problems


def check_behaviour(scenario: Scenario, record: TurnRecord, catalogue: Catalogue, known_revisions: set[str] | None = None) -> list[str]:
    """Single-turn form of :func:`check_turn` for the scenario's first message."""
    return check_turn(scenario.turns()[0], record, catalogue, set(known_revisions or ()))


# ----- records ---------------------------------------------------------

def training_messages(session: Session, records: list[TurnRecord]) -> list[dict[str, Any]]:
    """Transcript with ``train`` flags: turns whose calls were refused carry no loss."""
    decisions = {call.tool_call_id: call.decision for record in records for call in record.calls}
    messages = []
    for message in session.transcript():
        if message["role"] == "assistant":
            refused = any(decisions.get(call["id"]) == "rejected" for call in message.get("tool_calls") or [])
            message = {**message, "train": not refused}
        messages.append(message)
    return messages


def build_record(
    *,
    request: dict[str, Any],
    scenario: Scenario,
    session: Session,
    records: list[TurnRecord],
    catalogue: Catalogue,
    setup_log: list[dict[str, Any]],
    split: str,
    teacher: ModelAdapter,
    turn_problems: list[list[str]],
    tools_by_turn: list[list[str]],
) -> dict[str, Any]:
    calls = [call for record in records for call in record.calls]
    problems = [f"turn {index + 1}: {problem}" for index, items in enumerate(turn_problems) for problem in items]
    tool_results = [
        {
            "tool_call_id": call.tool_call_id,
            "tool": call.live_tool or call.tool,
            "decision": call.decision,
            "status": call.status,
            "executed": call.executed,
            "mocked": call.mocked,
        }
        for call in calls
    ]
    used = sorted({call.live_tool for call in calls if call.live_tool})
    usage = getattr(teacher, "usage", None)
    turns = scenario.turns()
    body = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "request_id": request["request_id"],
        "scenario_id": scenario.id,
        "behaviour": scenario.behaviour,
        "split": split,
        "catalogue_sha256": catalogue.sha256,
        "tool_versions": {name: tool_version(catalogue.get(name)) for name in used},
        "teacher": {"adapter": getattr(teacher, "name", "unknown"), "usage": usage.as_dict() if usage else None},
        "setup": setup_log,
        # One tool list for the whole conversation (a chat template renders it
        # once); tools_by_turn records what was offered at each user message.
        "tools": [view.as_function() for view in session.offered_views()],
        "tools_by_turn": tools_by_turn,
        "messages": training_messages(session, records),
        "turns": [
            {"behaviour": turns[index].behaviour, "outcome": record.outcome, "problems": turn_problems[index]}
            for index, record in enumerate(records)
        ],
        "tool_results": tool_results,
        "executed_calls": sum(1 for call in calls if call.executed),
        "mocked_calls": sum(1 for call in calls if call.mocked),
        "outcome": records[-1].outcome if records else "incomplete",
        "accepted": not problems and len(records) == len(turns),
        "problems": problems,
        "audit": [sanitize(record.as_dict()) for record in records],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body["record_id"] = hashlib.sha256(canonical_json({k: v for k, v in body.items() if k != "created_at"})).hexdigest()[:32]
    return body


async def run_request(
    request: dict[str, Any],
    scenario: Scenario,
    *,
    catalogue: Catalogue,
    transport: ToolTransport,
    teacher_factory: Callable[[], ModelAdapter],
    generation: GenerationLimits,
    split: str,
    limits: Limits | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> dict[str, Any]:
    project_id = await create_project(transport, catalogue, f"structllm-datagen {request['request_id']}")
    revision, setup_log = await run_setup(
        transport, catalogue, scenario, request.get("slots") or {},
        project_id=project_id, request_id=request["request_id"], generation=generation, sleep=sleep,
    )
    context = f"Current structure_revision_id: {revision}" if revision else None
    teacher = teacher_factory()
    session = Session(
        catalogue=catalogue,
        transport=transport,
        adapter=teacher,
        limits=limits,
        policy=generation_policy(catalogue, generation),
        project_id=project_id,
        context=context,
        session_id=f"datagen-{request['request_id']}",
        sleep=sleep,
    )
    known = {revision} if revision else set()
    texts = [request["text"], *(request.get("followups") or [])]
    records, turn_problems, tools_by_turn = [], [], []
    for turn, text in zip(scenario.turns(), texts):
        record = await session.send(text)
        records.append(record)
        tools_by_turn.append(list(record.offered_tools))
        turn_problems.append(check_turn(turn, record, catalogue, known))
        if record.outcome in FAILED_OUTCOMES:
            break  # later turns would answer a broken conversation
    return build_record(
        request=request, scenario=scenario, session=session, records=records,
        catalogue=catalogue, setup_log=setup_log, split=split, teacher=teacher,
        turn_problems=turn_problems, tools_by_turn=tools_by_turn,
    )


# ----- output ----------------------------------------------------------

def model_failure(record: dict[str, Any]) -> str | None:
    """The teacher's API error, if the conversation broke because the model call failed."""
    for audit in record.get("audit") or []:
        if audit.get("outcome") == "model_error":
            return "; ".join(audit.get("notes") or ["model error"])
    return None


class TrajectoryWriter:
    """Append-only JSONL output that can be resumed after an interruption."""

    FILES = ("accepted.jsonl", "rejected.jsonl", "errors.jsonl")

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def done(self) -> set[str]:
        """Requests with a final result; retryable errors are run again on resume."""
        ids = set()
        for name in self.FILES:
            path = self.directory / name
            if path.exists():
                with path.open(encoding="utf-8") as file:
                    for line in file:
                        if line.strip():
                            item = json.loads(line)
                            if not item.get("retryable"):
                                ids.add(item["request_id"])
        return ids

    def _append(self, name: str, item: dict[str, Any]) -> None:
        with (self.directory / name).open("a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    def add(self, record: dict[str, Any]) -> None:
        self._append("accepted.jsonl" if record["accepted"] else "rejected.jsonl", record)

    def error(self, request: dict[str, Any], message: str, *, retryable: bool = False) -> None:
        self._append("errors.jsonl", {"request_id": request["request_id"], "scenario_id": request["scenario_id"],
                                      "error": message[:1000], "retryable": retryable})

    def manifest(self, extra: dict[str, Any]) -> Path:
        counts: dict[str, Any] = {}
        totals = {"executed_calls": 0, "mocked_calls": 0}
        for name in self.FILES:
            path = self.directory / name
            rows = []
            if path.exists():
                with path.open(encoding="utf-8") as file:
                    rows = [json.loads(line) for line in file if line.strip()]
            counts[name] = len(rows)
            if name == "errors.jsonl":
                counts["retryable_errors"] = sum(1 for row in rows if row.get("retryable"))
            for row in rows:
                for key in totals:
                    totals[key] += row.get(key, 0)
            if name == "accepted.jsonl":
                counts["accepted_by_split"] = {
                    split: sum(1 for row in rows if row.get("split") == split) for split in ("train", "eval")
                }
        files = {
            name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
            for name in self.FILES
            if (self.directory / name).exists()
        }
        manifest = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "counts": counts,
            **totals,
            "files": files,
            "limitations": [
                "Acceptance is a structural check of conversation shape, not of scientific correctness.",
                "Mocked tool results are awaiting_approval envelopes created locally; the operation was never sent.",
                "Executed results come from a development server and may differ from production.",
            ],
            **extra,
        }
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path
