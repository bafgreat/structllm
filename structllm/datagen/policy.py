"""Which operations a data generator may actually run on the development server.

Decided from live metadata rather than a hard-coded tool list:

* a capability whose base price already reaches its own approval threshold
  (``estimated_cost.base_units >= approval_at_units``: GCMC, NEB, ion MD,
  voltage, breakthrough, flexible GCMC, ...) is never sent;
* a request naming an ML-potential or semi-empirical engine (engines the server
  reports with ``family: mlip``) is never sent;
* a request asking for a longer-than-standard run is never sent.

Those requests get a mock ``awaiting_approval`` envelope, recorded as mocked.
Everything else is cheap by StructSub's own pricing and is executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from structllm.orchestrator import ExecutionPolicy
from structllm.toolspec import Catalogue, capability_meta

ENGINE_KEYS = frozenset({"engine", "calculator", "potential", "backend", "model", "method_engine"})
LENGTH_KEYS = re.compile(r"(^|_)(steps|cycles|n_steps|production|equilibration|max_steps|nsteps|length)($|_)")
FALLBACK_ML_PATTERN = re.compile(r"mace|chgnet|m3gnet|sevenn|orb|mattersim|xtb|gfn|tblite|mlip|nequip|allegro", re.I)


@dataclass
class GenerationLimits:
    max_steps: int = 2000
    max_cycles: int = 20000
    ml_engines: frozenset[str] = field(default_factory=frozenset)
    simulate_tools: frozenset[str] = frozenset()  # extra operator choice
    expensive_tools: frozenset[str] = frozenset()  # derived from metadata


def expensive_from_metadata(catalogue: Catalogue) -> frozenset[str]:
    names = set()
    for name, tool in catalogue.tools.items():
        cost = capability_meta(tool).get("estimated_cost") or {}
        base, threshold = cost.get("base_units"), cost.get("approval_at_units")
        if isinstance(base, (int, float)) and isinstance(threshold, (int, float)) and base >= threshold:
            names.add(name)
        if capability_meta(tool).get("approval") == "always":
            names.add(name)
    return frozenset(names)


def ml_engines_from_listing(properties: dict[str, Any]) -> frozenset[str]:
    """Engine names in the 'mlip' family from a ``list_engines`` result."""
    names = set()
    for backend in properties.get("backends") or []:
        if isinstance(backend, dict) and backend.get("family") == "mlip" and isinstance(backend.get("name"), str):
            names.add(backend["name"].lower())
    return frozenset(names)


def _walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, path)
    else:
        yield path, value


def needs_simulated_approval(name: str, arguments: dict[str, Any], limits: GenerationLimits) -> bool:
    if name in limits.expensive_tools or name in limits.simulate_tools:
        return True
    nodes = arguments.get("nodes")
    if isinstance(nodes, list):
        # A workflow runs its nodes' capabilities, so each node is judged as if
        # it were called on its own. A node that cannot be read is not sent.
        for node in nodes:
            if not isinstance(node, dict) or not isinstance(node.get("capability"), str):
                return True
            parameters = node.get("parameters")
            if needs_simulated_approval(node["capability"], {"parameters": parameters if isinstance(parameters, dict) else {}}, limits):
                return True
    for path, value in _walk(arguments.get("parameters", {})):
        key = path.rsplit(".", 1)[-1].lower()
        if key in ENGINE_KEYS and isinstance(value, str):
            lowered = value.lower()
            if lowered in limits.ml_engines or FALLBACK_ML_PATTERN.search(lowered):
                return True
        if isinstance(value, (int, float)) and not isinstance(value, bool) and LENGTH_KEYS.search(key):
            cap = limits.max_cycles if "cycle" in key else limits.max_steps
            if value > cap:
                return True
    return False


def generation_policy(catalogue: Catalogue, limits: GenerationLimits) -> ExecutionPolicy:
    limits.expensive_tools = limits.expensive_tools | expensive_from_metadata(catalogue)
    return ExecutionPolicy(simulate_approval=lambda name, arguments: needs_simulated_approval(name, arguments, limits))
