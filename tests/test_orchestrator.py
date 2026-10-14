from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from structllm.adapters import ScriptedAdapter
from structllm.chat_format import ParsedTurn
from structllm.dataset import load_catalogue
from structllm.mcp_client import FakeTransport, model_view, sanitize
from structllm.orchestrator import ExecutionPolicy, Limits, Session
from structllm.retrieval import ToolRetriever, recall_at_k
from structllm.toolspec import (
    Catalogue,
    ToolCallError,
    build_view,
    coerce_arguments,
    prepare_call,
)

CATALOGUE = Path(__file__).resolve().parents[1] / "catalogues" / "mcp-catalogue-20261013T230056Z-3654c2ed5636.json"
PROJECT = "11111111-1111-4111-8111-111111111111"
REVISION = "22222222-2222-4222-8222-222222222222"
JOB = "33333333-3333-4333-8333-333333333333"
SECRET = "test-bearer-token-must-never-appear"


def catalogue() -> Catalogue:
    return Catalogue(load_catalogue(CATALOGUE))


def envelope(status: str, operation: str = "build_structure", **extra) -> dict:
    return {
        "schema_version": "1.0.0",
        "operation": operation,
        "operation_version": "1.0.0",
        "status": status,
        "job_id": JOB,
        "calculation_id": JOB,
        "project_id": PROJECT,
        "structure_revision_id": REVISION if status == "succeeded" else None,
        "properties": {"name": "ethanol"} if status == "succeeded" else {},
        "warnings": [],
        "errors": [],
        "value_origin": "backend_output" if status == "succeeded" else "unavailable",
        "provenance": {"input_hash": "abc", "user_id": "u-1", "workspace_id": "w-1", "worker": "x"},
        **extra,
    }


def call(name: str, arguments: dict, call_id: str = "c1") -> dict:
    return {"id": call_id, "name": name, "arguments": arguments}


def build_args(smiles: str = "CCO") -> dict:
    return {"project_id": PROJECT, "parameters": {"body": {"smiles": smiles}}}


def run(session: Session, text: str):
    return asyncio.run(session.send(text))


async def no_sleep(_seconds: float) -> None:
    return None


def make_session(turns, responses, *, limits=None, policy=None, cat=None) -> tuple[Session, FakeTransport, ScriptedAdapter]:
    cat = cat or catalogue()
    transport = FakeTransport(list(cat.tools.values()), responses)
    adapter = ScriptedAdapter(list(turns))
    session = Session(
        catalogue=cat,
        transport=transport,
        adapter=adapter,
        limits=limits or Limits(),
        policy=policy,
        project_id=PROJECT,
        session_id="session-1",
        sleep=no_sleep,
    )
    return session, transport, adapter


class ToolViewTests(unittest.TestCase):
    def test_view_hides_managed_fields_and_keeps_units(self) -> None:
        view = build_view(catalogue().get("relax"))
        parameters = view.as_function()["function"]["parameters"]
        self.assertNotIn("idempotency_key", parameters["properties"])
        self.assertNotIn("idempotency_key", parameters.get("required", []))
        self.assertIn("eV/Å", view.description)
        self.assertNotIn("Runs as a job", view.description)

    def test_compact_views_are_much_smaller_than_live_definitions(self) -> None:
        cat = catalogue()
        compact = sum(len(json.dumps(build_view(tool).as_function())) for tool in cat.tools.values())
        full = sum(len(json.dumps(tool)) for tool in cat.tools.values())
        self.assertLess(compact, full / 3)

    def test_coercion_decodes_objects_but_never_strings(self) -> None:
        schema = catalogue().get("build_structure")["inputSchema"]
        coerced = coerce_arguments(schema, {"project_id": PROJECT, "parameters": '{"body": {"smiles": "[1]"}}'})
        self.assertEqual(coerced["parameters"], {"body": {"smiles": "[1]"}})
        self.assertEqual(coerced["project_id"], PROJECT)

    def test_prepare_call_injects_deterministic_idempotency_key(self) -> None:
        cat = catalogue()
        arguments = {**build_args(), "idempotency_key": "model-chosen"}
        first = prepare_call(cat, name="build_structure", arguments=arguments, allowed={"build_structure"}, session_id="s", step=1)
        again = prepare_call(cat, name="build_structure", arguments=arguments, allowed={"build_structure"}, session_id="s", step=1)
        later = prepare_call(cat, name="build_structure", arguments=arguments, allowed={"build_structure"}, session_id="s", step=2)
        self.assertNotEqual(first["idempotency_key"], "model-chosen")
        self.assertEqual(first["idempotency_key"], again["idempotency_key"])
        self.assertNotEqual(first["idempotency_key"], later["idempotency_key"])

    def test_prepare_call_rejects_unknown_unoffered_and_invalid(self) -> None:
        cat = catalogue()
        with self.assertRaises(ToolCallError) as unknown:
            prepare_call(cat, name="run_shell", arguments={}, allowed={"run_shell"}, session_id="s", step=1)
        self.assertEqual(unknown.exception.code, "unknown_tool")
        with self.assertRaises(ToolCallError) as unoffered:
            prepare_call(cat, name="relax", arguments={}, allowed={"build_structure"}, session_id="s", step=1)
        self.assertEqual(unoffered.exception.code, "tool_not_offered")
        with self.assertRaises(ToolCallError) as invalid:
            prepare_call(
                cat,
                name="build_structure",
                arguments={"project_id": "not-a-uuid", "parameters": {"body": {"smiles": "CCO", "shell": "rm -rf /"}}},
                allowed={"build_structure"},
                session_id="s",
                step=1,
            )
        self.assertEqual(invalid.exception.code, "invalid_arguments")
        self.assertTrue(invalid.exception.details)

    def test_excluded_tools_disappear(self) -> None:
        cat = catalogue().without({"relax"})
        self.assertNotIn("relax", cat)
        self.assertIn("relax", cat.excluded)


class RetrievalTests(unittest.TestCase):
    def test_retrieval_is_deterministic_and_explained(self) -> None:
        retriever = ToolRetriever(catalogue())
        first = retriever.search("optimise my structure with UFF", k=5)
        second = retriever.search("optimise my structure with UFF", k=5)
        self.assertEqual([item.name for item in first], [item.name for item in second])
        self.assertEqual(first[0].name, "relax")
        self.assertIn("relax", first[0].explain())

    def test_recall_report_counts_denominators(self) -> None:
        retriever = ToolRetriever(catalogue())
        report = recall_at_k(retriever, [("simulate the powder XRD pattern", {"analyse"}), ("make a peptide", {"build_structure"})], k=6)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["target_tools"], 2)
        self.assertLessEqual(report["all_targets_retrieved"], 2)


class OrchestratorTests(unittest.TestCase):
    def test_success_path_offers_small_candidate_set(self) -> None:
        session, transport, adapter = make_session(
            [
                ParsedTurn("Build from SMILES.", "", [call("build_structure", build_args())]),
                ParsedTurn("", "Built ethanol as revision " + REVISION, []),
            ],
            {"build_structure": [envelope("succeeded")]},
        )
        record = run(session, "Build ethanol from the SMILES CCO")
        self.assertEqual(record.outcome, "completed")
        offered = adapter.seen[0][1]
        self.assertLessEqual(len(offered), Limits().retrieval_k + 2)
        self.assertIn("find_tools", offered)
        self.assertIn("build_structure", offered)
        self.assertEqual(len(transport.calls), 1)
        name, sent = transport.calls[0]
        self.assertEqual(name, "build_structure")
        self.assertTrue(sent["idempotency_key"].startswith("structllm-"))
        self.assertEqual(record.calls[0].status, "succeeded")
        self.assertTrue(record.calls[0].executed)

    def test_invalid_arguments_never_reach_server(self) -> None:
        session, transport, _ = make_session(
            [
                ParsedTurn("", "", [call("build_structure", {"project_id": PROJECT, "parameters": {"body": {"smiles": 5}}})]),
                ParsedTurn("", "The request was invalid.", []),
            ],
            {},
        )
        record = run(session, "Build ethanol from SMILES")
        self.assertEqual(transport.calls, [])
        self.assertEqual(record.calls[0].decision, "rejected")
        tool_message = json.loads(session.messages[-2]["content"])
        self.assertEqual(tool_message["status"], "rejected")
        self.assertFalse(tool_message["executed"])

    def test_unoffered_tool_needs_find_tools(self) -> None:
        session, transport, _ = make_session(
            [
                ParsedTurn("", "", [call("mechanics", {"project_id": PROJECT, "structure_revision_id": REVISION})]),
                ParsedTurn("", "", [call("find_tools", {"query": "elastic tensor and bulk modulus"}, "c2")]),
                ParsedTurn("", "", [call("mechanics", {"project_id": PROJECT, "structure_revision_id": REVISION}, "c3")]),
                ParsedTurn("", "Waiting for approval.", []),
            ],
            {"mechanics": [envelope("awaiting_approval", "mechanics")]},
        )
        record = run(session, "Build ethanol from the SMILES CCO")
        self.assertEqual(record.calls[0].error["code"], "tool_not_offered")
        self.assertIn("mechanics", record.calls[1].result["found"])
        self.assertEqual(record.calls[2].status, "awaiting_approval")
        self.assertEqual([name for name, _ in transport.calls], ["mechanics"])

    def test_awaiting_approval_is_never_polled_and_closes_tool_use(self) -> None:
        relax = {"project_id": PROJECT, "structure_revision_id": REVISION, "parameters": {"body": {"engine": "mace"}}}
        session, transport, _ = make_session(
            [
                ParsedTurn("", "", [call("relax", relax)]),
                ParsedTurn("", "", [call("relax", relax, "c2")]),
                ParsedTurn("", "This relaxation is waiting for approval.", []),
            ],
            {"relax": [envelope("awaiting_approval", "relax")]},
        )
        record = run(session, "optimise the structure with MACE")
        self.assertEqual(record.outcome, "awaiting_approval")
        self.assertEqual([name for name, _ in transport.calls], ["relax"])
        self.assertEqual(record.calls[1].error["code"], "awaiting_approval")
        view = json.loads(session.messages[3]["content"])
        self.assertIn("Do not resubmit", view["structllm_note"])

    def test_polling_is_bounded_and_pending_job_is_reported(self) -> None:
        limits = Limits(max_polls=3)
        session, transport, _ = make_session(
            [ParsedTurn("", "", [call("build_structure", build_args())]), ParsedTurn("", f"Job {JOB} is still running.", [])],
            {"build_structure": [envelope("queued")], "calculation_get": [envelope("running")]},
            limits=limits,
        )
        record = run(session, "Build ethanol from SMILES CCO")
        self.assertEqual(record.calls[0].polls, 3)
        self.assertEqual(record.calls[0].status, "running")
        self.assertEqual(record.outcome, "job_pending")
        self.assertEqual(len(transport.calls), 4)

    def test_failed_call_is_not_resubmitted_identically(self) -> None:
        failed = {"status": "failed", "error": {"code": "invalid_input", "message": "SMILES could not be parsed"}}
        bad = build_args("not-smiles")
        session, transport, _ = make_session(
            [
                ParsedTurn("", "", [call("build_structure", bad)]),
                ParsedTurn("", "", [call("build_structure", bad, "c2")]),
                ParsedTurn("", "The SMILES is invalid.", []),
            ],
            {"build_structure": [failed]},
        )
        record = run(session, "build not-smiles")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(record.calls[1].error["code"], "identical_resubmission_blocked")

    def test_deny_and_simulated_approval_policies(self) -> None:
        gcmc = {"project_id": PROJECT, "structure_revision_id": REVISION, "parameters": {"body": {}}}
        policy = ExecutionPolicy(deny=frozenset({"relax"}), simulate_approval=lambda name, _: name == "guest_gcmc")
        cat = catalogue()
        session, transport, _ = make_session(
            [
                ParsedTurn("", "", [call("find_tools", {"query": "grand canonical monte carlo adsorption"})]),
                ParsedTurn("", "", [call("guest_gcmc", gcmc, "c2")]),
                ParsedTurn("", "GCMC needs approval.", []),
            ],
            {},
            policy=policy,
            cat=cat,
        )
        record = run(session, "co2 uptake by gcmc")
        gcmc_record = record.calls[1]
        self.assertEqual(gcmc_record.decision, "simulated")
        self.assertTrue(gcmc_record.mocked)
        self.assertFalse(gcmc_record.executed)
        self.assertEqual(transport.calls, [])

    def test_call_budget_and_turn_limit(self) -> None:
        limits = Limits(max_tool_calls=1, max_model_turns=3)
        session, transport, _ = make_session(
            [
                ParsedTurn("", "", [call("build_structure", build_args()), call("build_structure", build_args("CC"), "c2")]),
                ParsedTurn("", "Done what was allowed.", []),
            ],
            {"build_structure": [envelope("succeeded")]},
            limits=limits,
        )
        record = run(session, "build ethanol and ethane")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(record.calls[1].error["code"], "tool_call_budget_exhausted")

    def test_records_and_transcripts_carry_no_credentials_or_identity(self) -> None:
        session, _, _ = make_session(
            [ParsedTurn("", "", [call("build_structure", build_args())]), ParsedTurn("", "Built.", [])],
            {"build_structure": [envelope("succeeded")]},
        )
        record = run(session, "Build ethanol from SMILES CCO")
        dumped = json.dumps([record.as_dict(), session.transcript()])
        for forbidden in (SECRET, "u-1", "w-1", '"worker"', "user_id", "workspace_id", "/control/approve"):
            self.assertNotIn(forbidden, dumped)

    def test_model_view_is_bounded(self) -> None:
        big = envelope("succeeded", properties={"rows": [{"x": index, "label": "y" * 300} for index in range(500)]})
        view = model_view(big, max_chars=3000)
        self.assertLessEqual(len(json.dumps(view)), 3000)
        self.assertEqual(view["status"], "succeeded")
        self.assertNotIn("user_id", json.dumps(sanitize(big)))


if __name__ == "__main__":
    unittest.main()
