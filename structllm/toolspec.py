"""Model-facing views of live MCP tools and validation of proposed calls.

The live catalogue is authoritative. A model never sees it whole: it is shown a
compact copy of a few retrieved tools (long schema prose trimmed, titles and
orchestrator-managed fields removed). Every proposed call is checked against the
full, unmodified live input schema before anything is sent to MCP.

Tool names, descriptions and schemas are untrusted server data. They are only
ever rendered as data inside a tool list; nothing here executes or evaluates them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from structllm.dataset import canonical_json, verify_catalogue

FORMAT_CHECKER = FormatChecker()

#: Fields the orchestrator supplies itself. The model is not shown them and any
#: value it proposes is replaced, so a generated key can never collide with, or
#: deliberately reuse, another operation's idempotency key.
MANAGED_FIELDS = ("idempotency_key",)

#: Tool names reserved for orchestrator-local tools; a live tool with one of
#: these names is refused rather than shadowed.
LOCAL_TOOL_NAMES = frozenset({"find_tools"})

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")
_JOB_NOTE = re.compile(
    r"\s*Runs as a job: the result is returned directly.*?(?=\n\S|\Z)", re.DOTALL
)
_APPROVAL_NOTE = re.compile(
    r"\s*Expensive requests \(ML potentials.*?do not resubmit them\.", re.DOTALL
)


class ToolCallError(ValueError):
    """A proposed call was refused before execution."""

    def __init__(self, code: str, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or []

    def as_result(self) -> dict[str, Any]:
        """The refusal as a tool result the model can read and correct from."""
        return {
            "status": "rejected",
            "executed": False,
            "error": {"code": self.code, "message": str(self), "details": self.details[:10]},
        }


def capability_meta(tool: dict[str, Any]) -> dict[str, Any]:
    meta = tool.get("_meta")
    if not isinstance(meta, dict):
        return {}
    value = meta.get("structsub/capability")
    return value if isinstance(value, dict) else {}


def tool_version(tool: dict[str, Any]) -> str | None:
    version = capability_meta(tool).get("version")
    return str(version) if version is not None else None


def schema_sha256(tool: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(tool.get("inputSchema", {}))).hexdigest()


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + " …"


def _compact_schema(schema: Any, *, description_chars: int, root: bool = False) -> Any:
    """Copy a JSON Schema for display: drop titles, shorten prose, collapse nullable unions.

    The result is still a JSON Schema, but it is only for the model to read; the
    original schema is always the one used for validation.
    """
    if isinstance(schema, list):
        return [_compact_schema(item, description_chars=description_chars) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"title", "examples", "$schema", "$id"}:
            continue
        if key == "description" and isinstance(value, str):
            # Pydantic copies the request model's docstring to the object schema;
            # property descriptions (which carry units) are the useful part.
            if root:
                continue
            result[key] = _trim(value, description_chars)
            continue
        if key == "properties" and isinstance(value, dict):
            result[key] = {
                name: _compact_schema(sub, description_chars=description_chars)
                for name, sub in value.items()
            }
            continue
        if key in {"anyOf", "oneOf"} and isinstance(value, list):
            branches = [branch for branch in value if not (isinstance(branch, dict) and branch.get("type") == "null")]
            if len(branches) == 1 and len(value) == 2 and isinstance(branches[0], dict):
                # Optional field: show the value type; null is still accepted on validation.
                merged = _compact_schema(branches[0], description_chars=description_chars)
                for extra_key, extra_value in result.items():
                    merged.setdefault(extra_key, extra_value)
                result.update(merged)
                continue
        result[key] = _compact_schema(value, description_chars=description_chars)
    if isinstance(result.get("type"), str) and result.get("type") == "object" and "properties" in result:
        # Request-model docstrings can repeat on nested objects; keep them short.
        if "description" in result and isinstance(result["description"], str):
            result["description"] = _trim(result["description"], description_chars)
    return result


def _remove_managed(schema: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name in MANAGED_FIELDS:
            properties.pop(name, None)
    if isinstance(schema.get("required"), list):
        schema["required"] = [name for name in schema["required"] if name not in MANAGED_FIELDS]
    return schema


def _approval_hint(meta: dict[str, Any]) -> str | None:
    approval = meta.get("approval")
    if approval == "always":
        return "Every request waits for a person's approval."
    if approval == "by_cost":
        return "Expensive requests (ML potentials, long runs, heavy simulations) wait for a person's approval."
    return None


@dataclass(frozen=True)
class ToolView:
    """One live tool as offered to a model, possibly under an alias."""

    name: str  # name the model uses
    live_name: str  # name the MCP server knows
    description: str
    parameters: dict[str, Any]
    version: str | None
    category: str

    def as_function(self) -> dict[str, Any]:
        """OpenAI/Hugging Face function-tool format, as chat templates expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def as_anthropic(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


@dataclass
class ViewOptions:
    description_chars: int = 900
    property_description_chars: int = 240
    include_units: bool = True


def build_view(
    tool: dict[str, Any],
    *,
    alias: str | None = None,
    description: str | None = None,
    options: ViewOptions | None = None,
) -> ToolView:
    options = options or ViewOptions()
    meta = capability_meta(tool)
    text = description or tool.get("description") or meta.get("description") or tool["name"]
    # The per-tool job/approval boilerplate is stated once in the system prompt.
    text = _APPROVAL_NOTE.sub("", _JOB_NOTE.sub("", text)).strip()
    text = _trim(text, options.description_chars)
    hint = _approval_hint(meta)
    if hint:
        text = f"{text}\n{hint}"
    units = meta.get("units")
    if options.include_units and isinstance(units, dict) and units:
        listed = ", ".join(
            f"{path.removeprefix('parameters.')}: {unit}" for path, unit in sorted(units.items())
        )
        text = f"{text}\nUnits: {_trim(listed, 300)}"
    schema = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {}
    parameters = _compact_schema(
        _remove_managed(schema), description_chars=options.property_description_chars, root=True
    )
    parameters.setdefault("type", "object")
    return ToolView(
        name=alias or tool["name"],
        live_name=tool["name"],
        description=text,
        parameters=parameters,
        version=tool_version(tool),
        category=str(meta.get("category") or "management"),
    )


FIND_TOOLS = ToolView(
    name="find_tools",
    live_name="find_tools",
    description=(
        "Search the StructSub capability catalogue when none of the offered tools fits the "
        "request. Returns the definitions of up to five more tools, which may then be called. "
        "Describe the operation in plain words (for example 'adsorption isotherm of CO2'). "
        "Read-only; runs no calculation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2, "maxLength": 300},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    version=None,
    category="local",
)


class Catalogue:
    """A verified snapshot of live tools with lookups used by the orchestrator."""

    def __init__(self, snapshot: dict[str, Any], *, exclude: set[str] | None = None):
        self.sha256 = verify_catalogue(snapshot)
        self.snapshot = snapshot
        excluded = exclude or set()
        tools = []
        for tool in snapshot["tools"]:
            name = tool["name"]
            if name in excluded:
                continue
            if name in LOCAL_TOOL_NAMES or not _NAME_PATTERN.match(name):
                # A server tool must never shadow an orchestrator tool or carry an
                # unrenderable name; it is left out rather than renamed.
                continue
            tools.append(tool)
        self.tools: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in tools}
        self.excluded = sorted(excluded & {tool["name"] for tool in snapshot["tools"]})
        self._validators: dict[str, Draft202012Validator] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def names(self) -> list[str]:
        return sorted(self.tools)

    def get(self, name: str) -> dict[str, Any]:
        return self.tools[name]

    def validator(self, name: str) -> Draft202012Validator:
        if name not in self._validators:
            schema = self.tools[name].get("inputSchema")
            if not isinstance(schema, dict):
                raise ToolCallError("schema_unavailable", f"Tool {name!r} has no input schema")
            Draft202012Validator.check_schema(schema)
            self._validators[name] = Draft202012Validator(schema, format_checker=FORMAT_CHECKER)
        return self._validators[name]

    def without(self, names: set[str]) -> Catalogue:
        """The same snapshot with tools removed, e.g. held-out or unavailable tools."""
        return Catalogue(self.snapshot, exclude=set(self.excluded) | names)

    def versions(self) -> dict[str, str | None]:
        return {name: tool_version(tool) for name, tool in sorted(self.tools.items())}


def schema_errors(validator: Draft202012Validator, value: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": [str(part) for part in error.absolute_path],
            "rule": str(error.validator),
            "message": error.message[:500],
        }
        for error in sorted(validator.iter_errors(value), key=lambda error: list(map(str, error.absolute_path)))
    ][:20]


def idempotency_key(session_id: str, step: int, tool_name: str, arguments: dict[str, Any]) -> str:
    """Deterministic per step, so a repeated delivery of one step never runs twice."""
    payload = canonical_json({"session": session_id, "step": step, "tool": tool_name, "arguments": arguments})
    return "structllm-" + hashlib.sha256(payload).hexdigest()[:40]


def _schema_type_allows_string(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return True
    kind = schema.get("type")
    if kind is None:
        branches = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(branches, list):
            return any(_schema_type_allows_string(branch) for branch in branches)
        return "enum" not in schema or any(isinstance(item, str) for item in schema["enum"])
    if isinstance(kind, list):
        return "string" in kind
    return kind == "string"


def coerce_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON text values where the schema expects a non-string.

    Some chat templates render every top-level parameter as text (Qwen3.5's XML
    tool-call format). A value is decoded only when its property does not accept
    a string, so a SMILES or name is never reinterpreted as JSON.
    """
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return arguments
    result = dict(arguments)
    for key, value in arguments.items():
        if isinstance(value, str) and key in properties and not _schema_type_allows_string(properties[key]):
            try:
                result[key] = json.loads(value)
            except (TypeError, ValueError):
                pass
    return result


def prepare_call(
    catalogue: Catalogue,
    *,
    name: str,
    arguments: Any,
    allowed: set[str],
    session_id: str,
    step: int,
) -> dict[str, Any]:
    """Validate a model-proposed call and return the exact MCP arguments.

    Refuses tools that were not offered to the model in this session, unknown or
    excluded tools, non-object arguments and anything failing the live schema.
    """
    if name not in allowed:
        if name in catalogue:
            raise ToolCallError(
                "tool_not_offered",
                f"Tool {name!r} was not offered in this conversation; use find_tools to look it up first.",
            )
        raise ToolCallError("unknown_tool", f"No tool named {name!r} is available.")
    if name not in catalogue:
        raise ToolCallError("unknown_tool", f"No tool named {name!r} is available.")
    if not isinstance(arguments, dict):
        raise ToolCallError("invalid_arguments", "Tool arguments must be a JSON object.")
    tool = catalogue.get(name)
    schema = tool.get("inputSchema", {})
    prepared = coerce_arguments(schema, arguments)
    for managed in MANAGED_FIELDS:
        prepared.pop(managed, None)
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if "idempotency_key" in properties:
        prepared["idempotency_key"] = idempotency_key(session_id, step, name, prepared)
    errors = schema_errors(catalogue.validator(name), prepared)
    if errors:
        raise ToolCallError(
            "invalid_arguments",
            "Arguments do not match the tool's input schema; nothing was executed.",
            errors,
        )
    return prepared
