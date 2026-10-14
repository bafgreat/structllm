from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from structllm.adapters import ScriptedAdapter
from structllm.chat_format import ParsedTurn
from structllm.datagen.policy import GenerationLimits, expensive_from_metadata, needs_simulated_approval
from structllm.datagen.requests import RequestWriter, check_request, generate_requests, writer_prompt
from structllm.datagen.scenarios import Scenario, ScenarioError, SetupStep, Turn, check_against_catalogue, load_scenarios
from structllm.datagen.trajectories import (
    SetupError,
    TrajectoryWriter,
    run_request,
    scenario_split,
)
from structllm.mcp_client import FakeTransport
from test_orchestrator import PROJECT, REVISION, catalogue, envelope, no_sleep

GCMC = Scenario(
    id="gcmc_uptake",
    behaviour="approval_hold",
    brief="GCMC of a gas in the current framework.",
    tools=("guest_gcmc",),
    setup=(SetupStep("build_structure", {"body": {"zeolite": "{zeolite}"}}),),
    slots={"zeolite": ("MFI",), "guest": ("CO2",)},
    variants=1,
)
BUILD = Scenario(id="build_molecule", behaviour="execute", brief="Build a molecule.", tools=("build_structure",),
                 slots={"molecule": ("ethanol",)}, variants=1)
CLARIFY = Scenario(id="ambiguous", behaviour="clarify", brief="Nothing identifies the structure.", variants=1)


def turn(content: str = "", *calls: dict) -> ParsedTurn:
    return ParsedTurn(reasoning="why", content=content, tool_calls=list(calls))


def request(scenario: Scenario, text: str, slots: dict | None = None) -> dict:
    return {"request_id": f"req-{scenario.id}", "scenario_id": scenario.id, "text": text,
            "slots": slots if slots is not None else {name: values[0] for name, values in scenario.slots.items()}}


def responses() -> dict:
    return {
        "project_create": [{"project_id": PROJECT, "name": "x"}],
        "build_structure": [envelope("succeeded")],
    }


def run(scenario: Scenario, req: dict, turns: list[ParsedTurn], transport_responses: dict | None = None):
    cat = catalogue()
    transport = FakeTransport(list(cat.tools.values()), transport_responses or responses())
    adapter = ScriptedAdapter(list(turns))
    record = asyncio.run(
        run_request(req, scenario, catalogue=cat, transport=transport, teacher_factory=lambda: adapter,
                    generation=GenerationLimits(ml_engines=frozenset({"mace"})), split="train", sleep=no_sleep)
    )
    return record, transport, adapter


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.cat = catalogue()
        self.limits = GenerationLimits(expensive_tools=expensive_from_metadata(self.cat), ml_engines=frozenset({"mace"}))

    def test_expensive_tools_come_from_server_pricing(self):
        expensive = expensive_from_metadata(self.cat)
        self.assertTrue({"guest_gcmc", "guest_breakthrough", "guest_flexible", "ion_voltage", "ion_hop_barrier"} <= expensive)
        self.assertNotIn("relax", expensive)
        self.assertNotIn("build_structure", expensive)

    def test_ml_engine_and_long_runs_are_not_sent(self):
        relax = lambda body: {"parameters": {"body": body}}  # noqa: E731
        self.assertFalse(needs_simulated_approval("relax", relax({"engine": "auto"}), self.limits))
        self.assertTrue(needs_simulated_approval("relax", relax({"engine": "MACE"}), self.limits))
        self.assertTrue(needs_simulated_approval("relax", relax({"engine": "gfn2-xtb"}), self.limits))
        self.assertTrue(needs_simulated_approval("relax", relax({"max_steps": 5000}), self.limits))
        self.assertFalse(needs_simulated_approval("relax", relax({"max_steps": 500}), self.limits))

    def test_workflow_nodes_are_judged_individually(self):
        def workflow(*nodes):
            return {"project_id": PROJECT, "nodes": list(nodes)}

        build = {"id": "a", "capability": "build_structure", "version": "1.0.0", "parameters": {"body": {"smiles": "CCO"}}}
        gcmc = {"id": "b", "capability": "guest_gcmc", "version": "1.0.0", "depends_on": ["a"]}
        ml = {"id": "c", "capability": "relax", "version": "1.0.0", "parameters": {"body": {"engine": "mace"}}}
        self.assertFalse(needs_simulated_approval("workflow_create", workflow(build), self.limits))
        self.assertTrue(needs_simulated_approval("workflow_create", workflow(build, gcmc), self.limits))
        self.assertTrue(needs_simulated_approval("workflow_create", workflow(build, ml), self.limits))
        self.assertTrue(needs_simulated_approval("workflow_create", workflow("not a node"), self.limits))


class ScenarioTests(unittest.TestCase):
    def test_packaged_scenarios_fit_the_catalogue(self):
        scenarios = load_scenarios()
        check_against_catalogue(scenarios, catalogue())
        self.assertEqual(len({scenario.id for scenario in scenarios}), len(scenarios))

    def test_assignments_are_deterministic_and_cover_values(self):
        scenario = Scenario(id="s", behaviour="execute", brief="b", tools=("build_structure",),
                            slots={"molecule": ("a", "b", "c")}, variants=3)
        self.assertEqual(scenario.assignments(5), scenario.assignments(5))
        self.assertEqual(sorted(item["molecule"] for item in scenario.assignments(5)), ["a", "b", "c"])

    def test_unknown_tool_is_refused(self):
        bad = Scenario(id="s", behaviour="execute", brief="b", tools=("no_such_tool",))
        with self.assertRaises(ScenarioError):
            check_against_catalogue([bad], catalogue())

    def test_zipped_slots_stay_paired(self):
        scenario = Scenario(id="z", behaviour="execute", brief="b", tools=("build_structure",), zip_slots=True,
                            slots={"bad": ("a1", "b1", "c1"), "fixed": ("a2", "b2", "c2")}, variants=3)
        for item in scenario.assignments(9):
            self.assertEqual(item["bad"][0], item["fixed"][0])

    def test_bad_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.yaml"
            path.write_text("version: 1\nscenarios:\n  - {id: x, behaviour: guess, brief: b}\n", encoding="utf-8")
            with self.assertRaises(ScenarioError):
                load_scenarios(path)


class FakeWriter:
    model = "fake"

    def __init__(self, texts):
        self.texts = texts

    def write(self, scenario, assignments):
        return self.texts[: len(assignments)]


class RequestTests(unittest.TestCase):
    def test_prompt_has_no_tool_names_or_setup_values(self):
        prompt = writer_prompt(GCMC, GCMC.assignments(1))
        self.assertNotIn("guest_gcmc", prompt)
        self.assertNotIn("MFI", prompt)  # setup-only slot: the user just says "this framework"
        self.assertIn("CO2", prompt)

    def test_checks(self):
        self.assertEqual(check_request(BUILD, {"molecule": "ethanol"}, "Please build Ethanol for me"), [])
        self.assertTrue(check_request(BUILD, {"molecule": "ethanol"}, "Please build methanol"))
        self.assertTrue(check_request(BUILD, {"molecule": "ethanol"}, "call build_structure on ethanol"))
        relax = Scenario(id="r", behaviour="execute", brief="b", tools=("relax",))
        self.assertEqual(check_request(relax, {}, "please relax my structure"), [])
        conditions = Scenario(id="c", behaviour="execute", brief="b", tools=("guest_gcmc",),
                              slots={"conditions": ("273 K, 0.1 to 10 bar",)})
        self.assertEqual(check_request(conditions, {"conditions": "273 K, 0.1 to 10 bar"}, "uptake at 273K from 0.1 up to 10 bar"), [])
        smiles = Scenario(id="m", behaviour="execute", brief="b", tools=("build_structure",), slots={"smiles": ("c1ccccc1O",)})
        self.assertTrue(check_request(smiles, {"smiles": "c1ccccc1O"}, "build C1CCCCC1O"))

    def test_generate_drops_bad_and_duplicate_requests(self):
        scenario = Scenario(id="s", behaviour="execute", brief="b", tools=("build_structure",),
                            slots={"molecule": ("ethanol",)}, variants=3)
        kept, dropped = generate_requests([scenario], FakeWriter([("build ethanol", []), ("Build  ETHANOL", []), None]), seed=1)
        self.assertEqual([item.text for item in kept], ["build ethanol"])
        self.assertEqual(len(dropped), 2)

    def test_anthropic_writer_uses_structured_output(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            payload = {"messages": [{"index": 0, "text": "Can you build ethanol?", "followups": []}]}
            return SimpleNamespace(
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            )

        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        writer = RequestWriter(client=client)
        self.assertEqual(writer.write(BUILD, BUILD.assignments(1)), [("Can you build ethanol?", [])])
        self.assertEqual(captured["model"], "claude-opus-5-5")
        self.assertEqual(captured["output_config"]["format"]["type"], "json_schema")


class TrajectoryTests(unittest.TestCase):
    def test_expensive_request_is_mocked_and_never_sent(self):
        gcmc_args = {"project_id": PROJECT, "structure_revision_id": REVISION,
                     "parameters": {"body": {"guests": ["co2"], "temperature": 298}}}
        record, transport, adapter = run(
            GCMC, request(GCMC, "What's the CO2 uptake of this at 298 K and 1 bar?"),
            [turn("", {"id": "t1", "name": "guest_gcmc", "arguments": gcmc_args}),
             turn("The GCMC simulation is waiting for approval in StructSub.")],
        )
        sent = [name for name, _ in transport.calls]
        self.assertEqual(sent, ["project_create", "build_structure"])
        self.assertTrue(record["accepted"], record["problems"])
        self.assertEqual((record["executed_calls"], record["mocked_calls"]), (0, 1))
        self.assertEqual(record["tool_results"][0]["mocked"], True)
        self.assertEqual(record["outcome"], "awaiting_approval")
        self.assertIn(f"Current structure_revision_id: {REVISION}", record["messages"][0]["content"])
        self.assertIn("guest_gcmc", [tool["function"]["name"] for tool in record["tools"]])
        self.assertTrue(all(message.get("train", True) for message in record["messages"]))

    def test_executed_call_and_rejected_turn_carries_no_loss(self):
        record, transport, _ = run(
            BUILD, request(BUILD, "build ethanol please"),
            [turn("", {"id": "t1", "name": "build_structure", "arguments": {"project_id": PROJECT, "parameters": {"body": {"smile": "CCO"}}}}),
             turn("", {"id": "t2", "name": "build_structure", "arguments": {"project_id": PROJECT, "parameters": {"body": {"smiles": "CCO"}}}}),
             turn("Built ethanol: revision 2222.")],
        )
        self.assertTrue(record["accepted"], record["problems"])
        self.assertEqual(record["executed_calls"], 1)
        assistant = [message for message in record["messages"] if message["role"] == "assistant"]
        self.assertEqual([message["train"] for message in assistant], [False, True, True])

    def test_clarify_must_ask_and_not_run(self):
        good, _, _ = run(CLARIFY, request(CLARIFY, "build the MOF from the paper"), [turn("Which MOF do you mean?")])
        self.assertTrue(good["accepted"], good["problems"])
        bad, _, _ = run(
            CLARIFY, request(CLARIFY, "build the MOF from the paper"),
            [turn("", {"id": "t1", "name": "build_structure", "arguments": {"project_id": PROJECT, "parameters": {"body": {"name": "MOF-5"}}}}),
             turn("I built MOF-5.")],
        )
        self.assertFalse(bad["accepted"])
        self.assertIn("turn 1: ran operations instead of asking", bad["problems"])

    def test_failed_setup_stops_before_the_teacher(self):
        failing = {"project_create": [{"project_id": PROJECT}], "build_structure": [envelope("failed")]}
        adapter_turns = [turn("should not run")]
        with self.assertRaises(SetupError):
            run(GCMC, request(GCMC, "x"), adapter_turns, failing)
        self.assertEqual(len(adapter_turns), 1)

    def test_writer_resumes_and_counts(self):
        record, _, _ = run(CLARIFY, request(CLARIFY, "build the thing"), [turn("Which structure?")])
        with tempfile.TemporaryDirectory() as directory:
            writer = TrajectoryWriter(Path(directory))
            writer.add(record)
            writer.error({"request_id": "r2", "scenario_id": "s"}, "boom")
            self.assertEqual(writer.done(), {record["request_id"], "r2"})
            manifest = json.loads(writer.manifest({"seed": 1}).read_text())
            self.assertEqual(manifest["counts"]["accepted.jsonl"], 1)
            self.assertEqual(manifest["counts"]["errors.jsonl"], 1)

    def test_followup_must_use_revision_from_the_conversation(self):
        scenario = Scenario(
            id="build_then_export", behaviour="execute", brief="b", tools=("build_structure",),
            followups=(Turn("execute", "export it", ("export_structure",), uses_prior_revision=True),), variants=1,
        )
        build = {"id": "t1", "name": "build_structure", "arguments": {"project_id": PROJECT, "parameters": {"body": {"smiles": "CCO"}}}}
        req = {**request(scenario, "build ethanol", {}), "followups": ["now give me it as xyz"]}

        def export(revision):
            return {"id": "t2", "name": "export_structure", "arguments": {
                "project_id": PROJECT, "structure_revision_id": revision, "parameters": {"query": {"format": "xyz"}}}}

        transport = {**responses(), "export_structure": [envelope("succeeded", operation="export_structure")]}
        good, _, _ = run(scenario, req, [turn("", build), turn("Built."), turn("", export(REVISION)), turn("Here is the xyz.")], transport)
        self.assertTrue(good["accepted"], good["problems"])
        self.assertEqual([item["behaviour"] for item in good["turns"]], ["execute", "execute"])
        self.assertEqual(sum(1 for message in good["messages"] if message["role"] == "user"), 2)
        invented = "44444444-4444-4444-8444-444444444444"
        transport = {**responses(), "export_structure": [envelope("succeeded", operation="export_structure")]}
        bad, _, _ = run(scenario, req, [turn("", build), turn("Built."), turn("", export(invented)), turn("Here is the xyz.")], transport)
        self.assertIn("turn 2: export_structure acted on a revision not from this conversation", bad["problems"])

    def test_forbidden_tool_fails_the_turn(self):
        scenario = Scenario(id="negated", behaviour="execute", brief="b", tools=("build_structure",),
                            forbidden_tools=("build_structure",), variants=1)
        build = {"id": "t1", "name": "build_structure", "arguments": {"project_id": PROJECT, "parameters": {"body": {"smiles": "CCO"}}}}
        record, _, _ = run(scenario, request(scenario, "please build ethanol", {}), [turn("", build), turn("done")])
        self.assertIn("turn 1: called build_structure, which this request rules out", record["problems"])

    def test_model_failures_are_retried_on_resume(self):
        from structllm.datagen.trajectories import model_failure

        record, _, _ = run(CLARIFY, request(CLARIFY, "build the thing"), [])  # adapter raises at once
        self.assertIn("no turns left", model_failure(record))
        with tempfile.TemporaryDirectory() as directory:
            writer = TrajectoryWriter(Path(directory))
            writer.error({"request_id": "r1", "scenario_id": "s"}, "credit balance is too low", retryable=True)
            writer.error({"request_id": "r2", "scenario_id": "s"}, "setup failed")
            self.assertEqual(writer.done(), {"r2"})

    def test_split_is_by_scenario(self):
        splits = {scenario_split(f"s{index}", seed=3, eval_fraction=0.3) for index in range(40)}
        self.assertEqual(splits, {"train", "eval"})
        self.assertEqual(scenario_split("x", seed=3, eval_fraction=0.3), scenario_split("x", seed=3, eval_fraction=0.3))


class ExportTests(unittest.TestCase):
    def test_export_is_self_contained_with_loss_flags(self):
        from structllm.datagen.export import export_sft

        record, _, _ = run(
            BUILD, request(BUILD, "build ethanol please"),
            [turn("", {"id": "t1", "name": "build_structure", "arguments": {"project_id": PROJECT, "parameters": {"body": {"smile": "CCO"}}}}),
             turn("", {"id": "t2", "name": "build_structure", "arguments": {"project_id": PROJECT, "parameters": {"body": {"smiles": "CCO"}}}}),
             turn("Built ethanol.")],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shard in ("shard-0-of-2", "shard-1-of-2"):
                TrajectoryWriter(root / "traj" / shard).add(record)  # duplicate is merged once
            destination = export_sft(root / "traj", root / "out")
            lines = (destination / "train.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            example = json.loads(lines[0])
            assistant = [m for m in example["messages"] if m["role"] == "assistant"]
            self.assertEqual([m["train"] for m in assistant], [False, True, True])
            self.assertEqual(assistant[1]["tool_calls"][0]["function"]["arguments"]["parameters"]["body"]["smiles"], "CCO")
            self.assertTrue(all(tool["type"] == "function" for tool in example["tools"]))
            self.assertNotIn("_provider_blocks", lines[0])
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["assistant_turns"]["train"], {"trained": 2, "context_only": 1})


if __name__ == "__main__":
    unittest.main()
