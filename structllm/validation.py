"""Run narrowly scoped, inexpensive live-MCP validation examples."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from jsonschema import Draft202012Validator, FormatChecker
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

from structllm.catalogue import discover_tools, sanitize_endpoint
from structllm.dataset import canonical_json

POLL_INTERVAL_SECONDS = 1
MAX_POLLS = 5


def _result_content(call_result: Any) -> dict[str, Any]:
    value = call_result.structured_content
    if isinstance(value, dict):
        return value
    raise RuntimeError("MCP tool response did not include structured content")


def _normalized_errors(result: dict[str, Any]) -> list[dict[str, str]]:
    errors = result.get("errors")
    if not isinstance(errors, list):
        error = result.get("error")
        errors = [error] if isinstance(error, dict) else []
    normalized = []
    for error in errors:
        if isinstance(error, dict):
            normalized.append(
                {
                    key: str(error[key])[:2000]
                    for key in ("code", "message")
                    if error.get(key) is not None
                }
            )
        elif error is not None:
            normalized.append({"message": str(error)[:2000]})
    return normalized


async def capture_invalid_smiles(access_token: str, endpoint: str) -> dict[str, Any]:
    """Create a local project, submit a known-invalid SMILES, and record its failure."""
    endpoint = sanitize_endpoint(endpoint)
    if not access_token:
        raise ValueError("MCP_ACCESS_TOKEN must be set")

    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {access_token}"}) as http:
        async with Client(
            streamable_http_client(endpoint, http_client=http), mode="auto"
        ) as client:
            tools = await discover_tools(client)
            definitions = {tool["name"]: tool for tool in tools}
            if "project_create" not in definitions or "build_structure" not in definitions:
                raise RuntimeError("Live catalogue must expose project_create and build_structure")

            project_schema = definitions["project_create"]["inputSchema"]
            project_arguments = {"name": f"StructLLM invalid-SMILES validation {uuid.uuid4().hex[:8]}"}
            Draft202012Validator(
                project_schema, format_checker=FormatChecker()
            ).validate(project_arguments)
            project_result = await client.call_tool("project_create", project_arguments)
            project_content = _result_content(project_result)
            if project_result.is_error or not isinstance(project_content.get("project_id"), str):
                raise RuntimeError("MCP project_create failed; no validation call was submitted")

            tool = definitions["build_structure"]
            metadata = tool.get("_meta", {}).get("structsub/capability", {})
            invalid_smiles = "this is not a valid SMILES"
            arguments = {
                "project_id": project_content["project_id"],
                "parameters": {"body": {"smiles": invalid_smiles}},
                "idempotency_key": uuid.uuid4().hex,
            }
            Draft202012Validator(
                tool["inputSchema"], format_checker=FormatChecker()
            ).validate(arguments)
            call_result = await client.call_tool("build_structure", arguments)
            result = _result_content(call_result)

            for poll_count in range(MAX_POLLS + 1):
                status = result.get("status")
                if status not in {"queued", "running"}:
                    break
                if poll_count == MAX_POLLS:
                    raise TimeoutError(
                        "Invalid-SMILES validation job did not finish within the poll limit"
                    )
                job_id = result.get("job_id")
                if not isinstance(job_id, str):
                    raise RuntimeError("In-progress MCP response has no job_id")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                polled = await client.call_tool("calculation_get", {"job_id": job_id})
                result = _result_content(polled)

            if result.get("status") != "failed":
                if result.get("status") == "awaiting_approval":
                    raise RuntimeError(
                        "Invalid-SMILES request unexpectedly requires approval; no approval was attempted"
                    )
                raise RuntimeError(
                    f"Expected a failed invalid-SMILES request; MCP returned status "
                    f"{result.get('status')!r}"
                )
            errors = _normalized_errors(result)
            if not errors:
                raise RuntimeError("MCP marked the request failed but returned no error details")

            return {
                "example_id": hashlib.sha256(
                    canonical_json(
                        {
                            "capability": "build_structure",
                            "version": metadata.get("version"),
                            "input": invalid_smiles,
                            "errors": errors,
                        }
                    )
                ).hexdigest(),
                "dataset_task": "observed_mcp_failure",
                "example_kind": "invalid_smiles_backend_failure",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Report tool failures honestly. Do not claim a structure was built "
                            "when the backend returned a failure."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Build a structure from this SMILES: this is not a valid SMILES",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "status": "failed",
                                "errors": errors,
                                "scientific_result": "unavailable",
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                "input": {"smiles": invalid_smiles},
                "observed_response": {"status": "failed", "errors": errors},
                "provenance": {
                    "source": "live_mcp_execution",
                    "endpoint": endpoint,
                    "tool_name": "build_structure",
                    "tool_version": metadata.get("version"),
                    "input_schema_sha256": hashlib.sha256(
                        canonical_json(tool["inputSchema"])
                    ).hexdigest(),
                    "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                },
                "validation": {
                    "input_schema_valid": True,
                    "backend_status": "failed",
                    "scientific_result_claimed": False,
                    "human_approval_used": False,
                    "expensive_operation": False,
                },
            }


def write_observed_failure(record: dict[str, Any], output_dir: Path) -> Path:
    """Save a sanitized failure example without overwriting existing evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    content = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    destination = output_dir / f"observed-failure-{digest[:16]}.jsonl"
    with destination.open("x", encoding="utf-8") as file:
        file.write(content)
    return destination


def capture_from_environment() -> dict[str, Any]:
    endpoint = os.environ.get("MCP_RESOURCE_URL")
    access_token = os.environ.get("MCP_ACCESS_TOKEN")
    if not endpoint:
        raise ValueError("Set MCP_RESOURCE_URL to the StructSub MCP endpoint")
    if not access_token:
        raise ValueError("Set MCP_ACCESS_TOKEN to an authorized bearer token")
    return asyncio.run(capture_invalid_smiles(access_token, endpoint))
