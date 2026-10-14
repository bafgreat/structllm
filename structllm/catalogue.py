"""Discover and snapshot tools exposed by an authenticated MCP server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

SNAPSHOT_SCHEMA_VERSION = "1.0.0"
MAX_CATALOGUE_PAGES = 1000


def _json_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected an MCP model or mapping, got {type(value).__name__}")


def sanitize_endpoint(endpoint: str) -> str:
    """Validate and return an MCP URL without user-info, query, or fragment."""
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname
    if (
        parsed.scheme not in {"https", "http"}
        or not hostname
        or (parsed.scheme == "http" and hostname not in {"localhost", "127.0.0.1", "::1"})
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/mcp")
    ):
        raise ValueError(
            "MCP_RESOURCE_URL must be an HTTPS MCP endpoint ending in /mcp; "
            "plain HTTP is allowed only for loopback development"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


async def discover_tools(client: Client) -> list[dict[str, Any]]:
    """Fetch every page of tools and preserve each server-provided definition."""
    tools: list[dict[str, Any]] = []
    cursors: set[str] = set()
    cursor: str | None = None

    for _ in range(MAX_CATALOGUE_PAGES):
        page = await client.list_tools(cursor=cursor, cache_mode="bypass")
        tools.extend(_json_model(tool) for tool in page.tools)
        cursor = page.next_cursor
        if cursor is None:
            break
        if cursor in cursors:
            raise RuntimeError("MCP server repeated a pagination cursor")
        cursors.add(cursor)
    else:
        raise RuntimeError(f"MCP catalogue exceeded the {MAX_CATALOGUE_PAGES}-page limit")

    names = [tool.get("name") for tool in tools]
    if len(names) != len(set(names)):
        raise RuntimeError("MCP catalogue contains duplicate tool names")
    return sorted(tools, key=lambda tool: tool["name"])


def build_snapshot(
    tools: list[dict[str, Any]],
    *,
    endpoint: str,
    server_info: Any = None,
    protocol_version: str | None = None,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic, versioned snapshot document from MCP responses."""
    ordered_tools = sorted(tools, key=lambda tool: tool["name"])
    names = [tool.get("name") for tool in ordered_tools]
    if len(names) != len(set(names)):
        raise ValueError("Cannot snapshot duplicate MCP tool names")

    catalogue_bytes = json.dumps(
        ordered_tools,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    timestamp = captured_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")

    snapshot: dict[str, Any] = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "captured_at": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "endpoint": sanitize_endpoint(endpoint),
        "catalogue_sha256": hashlib.sha256(catalogue_bytes).hexdigest(),
        "tool_count": len(ordered_tools),
        "tools": ordered_tools,
    }
    if server_info is not None:
        snapshot["server"] = _json_model(server_info)
    if protocol_version is not None:
        snapshot["protocol_version"] = protocol_version
    return snapshot


async def fetch_snapshot(
    endpoint: str,
    access_token: str,
    *,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Connect to the MCP endpoint and snapshot its authorized tool catalogue."""
    endpoint = sanitize_endpoint(endpoint)
    if not access_token:
        raise ValueError("MCP_ACCESS_TOKEN must be set")
    headers = {"Authorization": f"Bearer {access_token}"}
    if workspace_id:
        headers["x-structsub-workspace"] = workspace_id

    async with httpx2.AsyncClient(headers=headers) as http:
        async with Client(
            streamable_http_client(endpoint, http_client=http), mode="auto"
        ) as client:
            tools = await discover_tools(client)
            return build_snapshot(
                tools,
                endpoint=endpoint,
                server_info=client.server_info,
                protocol_version=client.session.protocol_version,
            )


def write_snapshot(snapshot: dict[str, Any], output_dir: Path) -> Path:
    """Write an immutable, timestamped JSON snapshot; do not overwrite files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.fromisoformat(snapshot["captured_at"].replace("Z", "+00:00"))
    stamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = snapshot["catalogue_sha256"][:12]
    destination = output_dir / f"mcp-catalogue-{stamp}-{digest}.json"
    with destination.open("x", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return destination


def snapshot_from_environment() -> dict[str, Any]:
    """Read credentials only from environment and retrieve the live catalogue."""
    endpoint = os.environ.get("MCP_RESOURCE_URL")
    access_token = os.environ.get("MCP_ACCESS_TOKEN")
    if not endpoint:
        raise ValueError("Set MCP_RESOURCE_URL to the StructSub MCP endpoint")
    if not access_token:
        raise ValueError("Set MCP_ACCESS_TOKEN to an authorized bearer token")
    return asyncio.run(
        fetch_snapshot(
            endpoint,
            access_token,
            workspace_id=os.environ.get("MCP_WORKSPACE_ID"),
        )
    )
