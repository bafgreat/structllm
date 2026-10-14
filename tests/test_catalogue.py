from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from mcp.types import Implementation, Tool

from structllm.catalogue import (
    build_snapshot,
    discover_tools,
    sanitize_endpoint,
    write_snapshot,
)
from structllm.dataset import (
    CatalogueError,
    example_value,
    generate_dataset,
    load_catalogue,
    verify_catalogue,
    write_dataset,
)
from structllm.validation import _normalized_errors, write_observed_failure


class CatalogueTests(unittest.TestCase):
    def test_snapshot_preserves_wire_schema_and_metadata(self) -> None:
        tool = Tool(
            name="sample_tool",
            description="A server-provided capability",
            input_schema={
                "type": "object",
                "properties": {"temperature": {"type": "number", "unit": "K"}},
                "required": ["temperature"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
            meta={"structsub/capability": {"id": "sample_tool", "version": "1.2.3"}},
        )

        snapshot = build_snapshot(
            [tool.model_dump(mode="json", by_alias=True, exclude_none=True)],
            endpoint="https://structsub.example/mcp",
            server_info=Implementation(name="StructSub", version="1.0.0"),
            protocol_version="2025-11-25",
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        )

        self.assertEqual(snapshot["snapshot_schema_version"], "1.0.0")
        self.assertEqual(snapshot["tool_count"], 1)
        self.assertEqual(snapshot["tools"][0]["inputSchema"]["properties"]["temperature"]["unit"], "K")
        self.assertEqual(snapshot["tools"][0]["_meta"]["structsub/capability"]["version"], "1.2.3")
        self.assertEqual(snapshot["server"]["name"], "StructSub")
        self.assertEqual(snapshot["captured_at"], "2026-01-02T00:00:00Z")

    def test_tool_order_does_not_change_catalogue_digest(self) -> None:
        tools = [
            {"name": "second", "inputSchema": {"type": "object"}},
            {"name": "first", "inputSchema": {"type": "object"}},
        ]
        first = build_snapshot(
            tools,
            endpoint="https://structsub.example/mcp",
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        second = build_snapshot(
            list(reversed(tools)),
            endpoint="https://structsub.example/mcp",
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        self.assertEqual(first["catalogue_sha256"], second["catalogue_sha256"])

    def test_rejects_duplicate_tool_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_snapshot(
                [{"name": "same"}, {"name": "same"}],
                endpoint="https://structsub.example/mcp",
            )

    def test_endpoint_validation_prevents_credential_persistence(self) -> None:
        with self.assertRaisesRegex(ValueError, "MCP endpoint"):
            sanitize_endpoint("https://user:secret@example.test/mcp?token=secret")
        with self.assertRaisesRegex(ValueError, "MCP endpoint"):
            sanitize_endpoint("http://remote.example/mcp")

    def test_snapshot_write_is_versioned_and_never_overwrites(self) -> None:
        snapshot = build_snapshot(
            [{"name": "sample"}],
            endpoint="https://structsub.example/mcp",
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_snapshot(snapshot, Path(directory))
            self.assertEqual(json.loads(path.read_text())["catalogue_sha256"], snapshot["catalogue_sha256"])
            with self.assertRaises(FileExistsError):
                write_snapshot(snapshot, Path(directory))


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_reads_all_pages_and_sorts_tools(self) -> None:
        class FakeClient:
            async def list_tools(self, *, cursor, cache_mode):
                self_cache_mode = cache_mode
                if self_cache_mode != "bypass":
                    raise AssertionError("Discovery must not use a stale tools/list cache")
                if cursor is None:
                    return SimpleNamespace(
                        tools=[{"name": "z_tool", "inputSchema": {"type": "object"}}],
                        next_cursor="page-2",
                    )
                if cursor == "page-2":
                    return SimpleNamespace(
                        tools=[{"name": "a_tool", "inputSchema": {"type": "object"}}],
                        next_cursor=None,
                    )
                raise AssertionError(f"Unexpected cursor: {cursor}")

        self.assertEqual(
            [tool["name"] for tool in await discover_tools(FakeClient())],
            ["a_tool", "z_tool"],
        )

    async def test_discovery_rejects_repeated_pagination_cursor(self) -> None:
        class RepeatingCursorClient:
            async def list_tools(self, *, cursor, cache_mode):
                return SimpleNamespace(tools=[], next_cursor="same")

        with self.assertRaisesRegex(RuntimeError, "repeated"):
            await discover_tools(RepeatingCursorClient())


class DatasetTests(unittest.TestCase):
    def make_snapshot(self) -> dict:
        tools = [
            {
                "name": "build_structure",
                "title": "Build structure",
                "description": "Create a structure from a specification.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "format": "uuid"},
                        "parameters": {
                            "type": "object",
                            "properties": {"formula": {"type": "string"}},
                            "required": ["formula"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["project_id", "parameters"],
                    "additionalProperties": False,
                },
                "_meta": {"structsub/capability": {"category": "build", "version": "1.0.0"}},
            },
            {
                "name": "list_structures",
                "title": "List structures",
                "description": "List available structures.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 1, "default": 5}},
                    "additionalProperties": False,
                },
                "_meta": {"structsub/capability": {"category": "build", "version": "1.0.0"}},
            },
            {
                "name": "relax_structure",
                "title": "Relax",
                "description": "Relax an existing structure.",
                "inputSchema": {
                    "type": "object",
                    "$defs": {
                        "Node": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,39}$"},
                                "capability": {"type": "string"},
                            },
                            "required": ["id", "capability"],
                        }
                    },
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"$ref": "#/$defs/Node"},
                        },
                        "steps": {"type": "integer", "exclusiveMinimum": 0, "maximum": 4},
                    },
                    "required": ["nodes", "steps"],
                    "additionalProperties": False,
                },
                "_meta": {"structsub/capability": {"category": "analysis", "version": "2.1.0"}},
            },
            {
                "name": "project_create",
                "description": "Create a project.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "minLength": 1}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        ]
        return build_snapshot(
            tools,
            endpoint="http://127.0.0.1/mcp",
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        )

    def test_examples_validate_schemas_and_group_holdout_tools(self) -> None:
        snapshot = self.make_snapshot()
        splits, quarantined, report = generate_dataset(snapshot, seed=17, holdout_fraction=0.25)
        self.assertEqual(quarantined, [])
        self.assertEqual(report["tool_count"], 4)
        self.assertEqual(report["validation_counts"]["schema_examples_validated"], 4)
        self.assertEqual(report["validation_counts"]["clarification_examples_validated"], 3)
        train_names = {record["provenance"]["tool_name"] for record in splits["train"]}
        heldout_names = {
            record["provenance"]["tool_name"] for record in splits["heldout_tools"]
        }
        self.assertTrue(train_names)
        self.assertTrue(heldout_names)
        self.assertFalse(train_names & heldout_names)
        for record in splits["train"] + splits["heldout_tools"]:
            self.assertFalse(record["validation"]["executable"])
            self.assertEqual(record["validation"]["scientific_validity"], "not_assessed")
            self.assertIn("untrusted data", record["messages"][0]["content"])
            self.assertTrue(
                all(candidate["name"] not in (
                    heldout_names if record["split"] == "train" else train_names
                ) for candidate in record["available_tools"])
            )
            if record["example_kind"] == "missing_arguments":
                self.assertEqual(record["validation"]["status"], "expected_schema_rejection")

    def test_catalogue_integrity_is_checked_before_generation(self) -> None:
        snapshot = self.make_snapshot()
        snapshot["tools"][0]["description"] = "tampered"
        with self.assertRaisesRegex(CatalogueError, "SHA-256"):
            verify_catalogue(snapshot)

    def test_schema_example_supports_refs_and_constrained_values(self) -> None:
        schema = {
            "type": "object",
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"key": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,39}$"}},
                    "required": ["key"],
                }
            },
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "items": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/Item"}},
                "count": {"type": "integer", "exclusiveMinimum": 0, "maximum": 3},
            },
            "required": ["id", "items", "count"],
        }
        value = example_value(schema)
        self.assertEqual(_schema_errors(schema, value), [])

    def test_dataset_files_are_versioned_and_non_overwriting(self) -> None:
        snapshot = self.make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = write_dataset(snapshot, output_dir=Path(directory), seed=17)
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["source_catalogue_sha256"], snapshot["catalogue_sha256"])
            self.assertEqual(manifest["scientific_execution"], "none")
            self.assertIn("train.jsonl", manifest["files"])
            self.assertTrue((path / "heldout-tools.jsonl").exists())
            with self.assertRaises(FileExistsError):
                write_dataset(snapshot, output_dir=Path(directory), seed=17)

    def test_catalogue_loader_checks_content_hash(self) -> None:
        snapshot = self.make_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            self.assertEqual(load_catalogue(path)["catalogue_sha256"], snapshot["catalogue_sha256"])


class LiveValidationRecordTests(unittest.TestCase):
    def test_normalizes_real_error_envelopes_without_result_ids(self) -> None:
        self.assertEqual(
            _normalized_errors(
                {
                    "errors": [
                        {"code": "invalid_smiles", "message": "Could not parse structure", "job_id": "ignored"}
                    ]
                }
            ),
            [{"code": "invalid_smiles", "message": "Could not parse structure"}],
        )
        self.assertEqual(
            _normalized_errors({"error": {"code": "invalid_input", "message": "Bad request"}}),
            [{"code": "invalid_input", "message": "Bad request"}],
        )

    def test_failure_record_write_does_not_overwrite(self) -> None:
        record = {"example_kind": "invalid_smiles_backend_failure"}
        with tempfile.TemporaryDirectory() as directory:
            path = write_observed_failure(record, Path(directory))
            self.assertEqual(json.loads(path.read_text()), record)
            with self.assertRaises(FileExistsError):
                write_observed_failure(record, Path(directory))


def _schema_errors(schema: dict, value: object) -> list:
    from jsonschema import Draft202012Validator, FormatChecker

    return list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value))


if __name__ == "__main__":
    unittest.main()
