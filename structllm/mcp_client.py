"""Bounded access to a StructSub MCP server through the official SDK client.

Everything returned by the server is untrusted data. Results are sanitised
(caller identity removed) before they are stored or shown to a model, and the
model-facing copy is size-bounded so one large report cannot crowd out the
conversation.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

from structllm.catalogue import build_snapshot, discover_tools, sanitize_endpoint

#: Provenance keys identifying a person, workspace or machine; never stored.
IDENTITY_KEYS = frozenset({"user_id", "workspace_id", "worker", "approval_user_id", "approved_by"})
#: Envelope keys that only repeat other fields; dropped from the model's copy.
REDUNDANT_KEYS = frozenset({"calculation_id", "result_id", "schema_version", "structure_version"})
TEXT_MEDIA_TYPES = ("application/json", "text/", "chemical/", "application/xml")


class ToolTransport(Protocol):
    """What the orchestrator needs from an MCP connection."""

    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def sanitize(value: Any) -> Any:
    """Recursively drop identity fields from a server result."""
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key not in IDENTITY_KEYS}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def decode_artifact(result: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    """Replace an artifact chunk's base64 with text when the media type is textual."""
    if not isinstance(result, dict) or "data_base64" not in result:
        return result
    media_type = str(result.get("media_type") or "")
    decoded = dict(result)
    data = decoded.pop("data_base64")
    if not media_type.startswith(TEXT_MEDIA_TYPES):
        decoded["data"] = f"<{len(data)} base64 characters of {media_type or 'binary'} data omitted>"
        return decoded
    try:
        text = base64.b64decode(data, validate=True).decode("utf-8", errors="replace")
    except (binascii.Error, ValueError):
        decoded["data"] = "<undecodable artifact chunk omitted>"
        return decoded
    decoded["text"] = text[:max_chars]
    if len(text) > max_chars:
        decoded["text_truncated"] = True
        offset = int(result.get("offset") or 0)
        # Bytes and characters differ for non-ASCII text; this is the conservative
        # offset of the shown prefix for the next request.
        decoded["next_offset"] = offset + len(text[:max_chars].encode("utf-8"))
    return decoded


def bound_json(value: Any, max_chars: int, *, max_items: int = 25, max_string: int = 1500) -> Any:
    """Shrink a JSON value until its serialisation fits ``max_chars``.

    Lists keep their first items and record how many were dropped; long strings
    are cut with a marker. Status, IDs, warnings and errors are top-level fields
    of the envelope, so they survive the progressive shrinking.
    """

    def shrink(item: Any, items: int, string: int, depth: int = 0) -> Any:
        if isinstance(item, dict):
            if depth > 8:
                return "<nested object omitted>"
            return {key: shrink(sub, items, string, depth + 1) for key, sub in item.items()}
        if isinstance(item, list):
            head = [shrink(sub, items, string, depth + 1) for sub in item[:items]]
            if len(item) > items:
                head.append(f"<{len(item) - items} more items omitted>")
            return head
        if isinstance(item, str) and len(item) > string:
            return item[:string] + f"<{len(item) - string} more characters omitted>"
        if isinstance(item, float):
            return float(f"{item:.6g}")
        return item

    current = shrink(value, max_items, max_string)
    while len(json.dumps(current, ensure_ascii=False)) > max_chars and (max_items > 2 or max_string > 80):
        max_items = max(2, max_items // 2)
        max_string = max(80, max_string // 2)
        current = shrink(value, max_items, max_string)
    if len(json.dumps(current, ensure_ascii=False)) > max_chars and isinstance(current, dict):
        # Keep the envelope essentials and drop bulky detail fields.
        essentials = {
            key: current[key]
            for key in (
                "operation",
                "status",
                "job_id",
                "project_id",
                "structure_revision_id",
                "structure_id",
                "workflow_id",
                "warnings",
                "errors",
                "error",
                "units",
                "value_origin",
                "artifact_ids",
            )
            if key in current
        }
        essentials["truncated"] = "result too large to show; details are in the stored record or artifacts"
        current = essentials
    return current


def model_view(result: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    """The sanitised, bounded copy of a tool result that a model reads."""
    view = copy.deepcopy(sanitize(result))
    if isinstance(view, dict):
        for key in REDUNDANT_KEYS:
            view.pop(key, None)
        provenance = view.get("provenance")
        if isinstance(provenance, dict):
            view["provenance"] = {
                key: provenance[key]
                for key in ("input_hash", "approval_required", "executed_at", "original_revision_id", "progress")
                if provenance.get(key) is not None
            }
        view = decode_artifact(view, max_chars=max_chars // 2)
    return bound_json(view, max_chars)


@dataclass
class MCPSettings:
    endpoint: str
    access_token: str
    workspace_id: str | None = None
    call_timeout_seconds: float = 90.0
    connect_timeout_seconds: float = 15.0

    @classmethod
    def from_environment(cls) -> MCPSettings:
        endpoint = os.environ.get("MCP_RESOURCE_URL")
        token = os.environ.get("MCP_ACCESS_TOKEN")
        if not endpoint or not token:
            raise ValueError("Set MCP_RESOURCE_URL and MCP_ACCESS_TOKEN (never pass tokens on the command line)")
        timeout = float(os.environ.get("STRUCTLLM_MCP_CALL_TIMEOUT", "90"))
        return cls(sanitize_endpoint(endpoint), token, os.environ.get("MCP_WORKSPACE_ID"), timeout)

    def __repr__(self) -> str:  # never print the bearer token
        return f"MCPSettings(endpoint={self.endpoint!r}, workspace_id={self.workspace_id!r})"


class MCPConnection:
    """An open MCP client session; use ``async with MCPConnection(settings)``."""

    def __init__(self, settings: MCPSettings):
        self.settings = settings
        self._stack: contextlib.AsyncExitStack | None = None
        self.client: Client | None = None

    async def __aenter__(self) -> MCPConnection:
        headers = {"Authorization": f"Bearer {self.settings.access_token}"}
        if self.settings.workspace_id:
            headers["x-structsub-workspace"] = self.settings.workspace_id
        self._stack = contextlib.AsyncExitStack()
        http = await self._stack.enter_async_context(
            httpx2.AsyncClient(
                headers=headers,
                timeout=httpx2.Timeout(self.settings.call_timeout_seconds, connect=self.settings.connect_timeout_seconds),
            )
        )
        self.client = await self._stack.enter_async_context(
            Client(streamable_http_client(sanitize_endpoint(self.settings.endpoint), http_client=http), mode="auto")
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def list_tools(self) -> list[dict[str, Any]]:
        assert self.client is not None
        return await discover_tools(self.client)

    async def snapshot(self) -> dict[str, Any]:
        assert self.client is not None
        tools = await discover_tools(self.client)
        return build_snapshot(
            tools,
            endpoint=self.settings.endpoint,
            server_info=self.client.server_info,
            protocol_version=self.client.session.protocol_version,
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        try:
            result = await asyncio.wait_for(
                self.client.call_tool(name, arguments), timeout=self.settings.call_timeout_seconds
            )
        except TimeoutError:
            # The request may still have been accepted; the caller must not rerun it.
            return {
                "status": "unknown",
                "error": {
                    "code": "client_timeout",
                    "message": "No response within the client time limit; the request may still be running. It was not resubmitted.",
                },
            }
        content = result.structured_content
        if not isinstance(content, dict):
            texts = [getattr(item, "text", "") for item in result.content or []]
            try:
                content = json.loads("".join(texts))
            except ValueError:
                content = {"status": "failed", "error": {"code": "unstructured_response", "message": "".join(texts)[:2000]}}
        if result.is_error and "status" not in content:
            content = {"status": "failed", **content}
        return content


class FakeTransport:
    """In-memory MCP stand-in for tests: scripted responses per tool name.

    ``responses[name]`` is a list consumed in order (the last one repeats) or a
    callable ``(arguments) -> result``. Every call is recorded.
    """

    def __init__(self, tools: list[dict[str, Any]], responses: dict[str, Any] | None = None):
        self.tools = tools
        self.responses = responses or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, copy.deepcopy(arguments)))
        scripted = self.responses.get(name)
        if callable(scripted):
            return scripted(arguments)
        if isinstance(scripted, list) and scripted:
            return copy.deepcopy(scripted.pop(0) if len(scripted) > 1 else scripted[0])
        return {"status": "failed", "error": {"code": "not_scripted", "message": f"No fake response for {name}"}}
