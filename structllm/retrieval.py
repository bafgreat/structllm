"""Explainable lexical retrieval of candidate tools for one request.

A BM25 index over the live tool name, description, capability metadata and
schema property names. A short, reviewable synonym table bridges common
researcher wording ("optimise", "surface area", "uptake") to the vocabulary
the tool descriptions use. No embedding service or model is involved, so a
ranking can be explained term by term and reproduced exactly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from structllm.toolspec import Catalogue, capability_meta

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """a an and are as at be by can do for from has have how i in into is it its
    me my of on or our please so that the their them then this to was we what
    when which will with would you your one each any all only new using use""".split()
)

#: Researcher wording -> catalogue wording. Expansions are added at a reduced
#: weight, so an exact term in the request still dominates.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "optimise": ("relax", "optimisation", "geometry"),
    "optimize": ("relax", "optimisation", "geometry"),
    "optimisation": ("relax",),
    "optimization": ("relax", "optimisation"),
    "minimise": ("relax",),
    "minimize": ("relax",),
    "relaxation": ("relax",),
    "mof": ("framework", "frameworks", "metal", "organic"),
    "mofs": ("framework", "frameworks"),
    "cof": ("framework", "frameworks", "covalent"),
    "cofs": ("framework", "frameworks"),
    "linker": ("linkers", "building", "unit", "substitute"),
    "linkers": ("linker", "building", "unit"),
    "node": ("sbu", "building", "unit"),
    "sbu": ("building", "unit", "node", "sbus"),
    "topology": ("net", "topologies", "rcsr"),
    "topologies": ("net", "topology"),
    "porosity": ("pore", "surface", "volume", "analyse"),
    "surface": ("porosity", "area"),
    "pore": ("porosity",),
    "pores": ("porosity", "pore", "guests"),
    "uptake": ("adsorption", "isotherm", "gcmc"),
    "adsorption": ("isotherm", "gcmc", "uptake"),
    "adsorb": ("adsorption", "isotherm"),
    "isotherm": ("adsorption", "gcmc"),
    "co2": ("guest", "adsorption", "gas"),
    "gas": ("guest", "adsorption"),
    "solvent": ("guests", "remove"),
    "guest": ("guests",),
    "smiles": ("molecule", "structure", "build"),
    "molecule": ("smiles", "structure"),
    "make": ("build", "create"),
    "construct": ("build", "assemble"),
    "generate": ("build", "create"),
    "create": ("build",),
    "replace": ("substitute", "substitution", "replace"),
    "swap": ("substitute", "replace"),
    "substitution": ("substitute",),
    "functionalise": ("functional", "group", "functionalise"),
    "functionalize": ("functionalise", "functional", "group"),
    "amino": ("functionalise", "functional", "group", "nh2"),
    "doping": ("replace", "atom", "substitute"),
    "dope": ("replace", "atom"),
    "metal": ("replace", "node", "sbu"),
    "hydrogens": ("hydrogen", "add_hydrogens"),
    "hydrogen": ("hydrogens",),
    "xrd": ("pxrd", "diffraction", "powder", "simulated"),
    "simulate": ("simulated", "analyse"),
    "characterise": ("analyse", "characterise"),
    "characterize": ("analyse", "characterise"),
    "analyze": ("analyse",),
    "bet": ("porosity", "surface", "area"),
    "pxrd": ("diffraction", "powder", "pattern"),
    "diffraction": ("pxrd",),
    "cif": ("import", "file", "export"),
    "upload": ("import",),
    "download": ("export",),
    "save": ("export",),
    "battery": ("ion", "voltage", "migration", "mobile"),
    "lithium": ("ion", "mobile", "li"),
    "li": ("ion", "mobile"),
    "sodium": ("ion", "mobile"),
    "conductivity": ("ion", "diffusion"),
    "diffusivity": ("diffusion",),
    "elastic": ("mechanics", "moduli", "tensor"),
    "bulk": ("mechanics", "moduli"),
    "modulus": ("mechanics", "moduli"),
    "stiffness": ("mechanics", "elastic"),
    "zeolite": ("zeolites", "framework", "type"),
    "zeolites": ("zeolite",),
    "polymer": ("chain", "polymer", "periodic"),
    "peptide": ("sequence", "amino", "residue", "monomers"),
    "protein": ("biomolecule", "peptide"),
    "crystal": ("space", "group", "prototype", "structure"),
    "perovskite": ("prototype", "structure", "type"),
    "undo": ("undo", "revert"),
    "revert": ("undo",),
    "status": ("calculation_get", "retrieve"),
    "job": ("calculation", "status"),
    "cancel": ("calculation_cancel", "cancel"),
    "selectivity": ("separates", "mixture", "iast"),
    "separation": ("selectivity", "breakthrough", "mixture"),
    "breathing": ("flexible",),
    "stacking": ("layers", "stack", "aa", "ab"),
    "layer": ("layers", "stack"),
    "bond": ("bonds",),
    "cite": ("citation",),
    "reference": ("citation",),
    "name": ("names", "lookup"),
    "identify": ("lookup", "find"),
    "dock": ("binding", "guest", "site"),
    "binding": ("dock", "site"),
}

FIELD_WEIGHTS = {"name": 3.0, "description": 1.0, "meta": 1.0, "properties": 0.5, "category": 1.0}


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(text.lower().replace("_", " ")):
        if token in _STOP or len(token) < 2 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _property_text(schema: Any, depth: int = 0) -> list[str]:
    if depth > 6 or not isinstance(schema, dict):
        return []
    parts: list[str] = []
    for name, sub in (schema.get("properties") or {}).items():
        if name in {"parameters", "body", "query", "path", "form"}:
            parts.extend(_property_text(sub, depth + 1))
            continue
        parts.append(name)
        if isinstance(sub, dict):
            if isinstance(sub.get("description"), str):
                parts.append(sub["description"][:300])
            parts.extend(_property_text(sub, depth + 1))
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(key) or []:
            parts.extend(_property_text(branch, depth + 1))
    if isinstance(schema.get("items"), dict):
        parts.extend(_property_text(schema["items"], depth + 1))
    return parts


def tool_fields(tool: dict[str, Any]) -> dict[str, list[str]]:
    meta = capability_meta(tool)
    return {
        "name": tokenize(tool["name"]),
        "description": tokenize(tool.get("description") or ""),
        "meta": tokenize(" ".join(str(meta.get(key) or "") for key in ("name", "description"))),
        "properties": tokenize(" ".join(_property_text(tool.get("inputSchema") or {}))),
        "category": tokenize(str(meta.get("category") or "")),
    }


@dataclass
class Ranked:
    name: str
    score: float
    matched: dict[str, float] = field(default_factory=dict)  # term -> contribution

    def explain(self, limit: int = 6) -> str:
        top = sorted(self.matched.items(), key=lambda item: -item[1])[:limit]
        return ", ".join(f"{term}={value:.2f}" for term, value in top)


class ToolRetriever:
    """BM25 over weighted tool fields; deterministic and explainable."""

    def __init__(self, catalogue: Catalogue, *, k1: float = 1.2, b: float = 0.75, synonym_weight: float = 0.4):
        self.catalogue = catalogue
        self.k1, self.b, self.synonym_weight = k1, b, synonym_weight
        self.docs: dict[str, Counter[str]] = {}
        for name, tool in catalogue.tools.items():
            counts: Counter[str] = Counter()
            for field_name, tokens in tool_fields(tool).items():
                for token in tokens:
                    counts[token] += FIELD_WEIGHTS[field_name]
            self.docs[name] = counts
        self.lengths = {name: sum(counts.values()) for name, counts in self.docs.items()}
        self.average = (sum(self.lengths.values()) / len(self.lengths)) if self.lengths else 1.0
        frequency: Counter[str] = Counter()
        for counts in self.docs.values():
            frequency.update(set(counts))
        total = len(self.docs)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in frequency.items()
        }

    def query_terms(self, text: str) -> dict[str, float]:
        terms: dict[str, float] = {}
        for token in tokenize(text):
            terms[token] = max(terms.get(token, 0.0), 1.0)
            for synonym in SYNONYMS.get(token, ()):
                for part in tokenize(synonym):
                    terms[part] = max(terms.get(part, 0.0), self.synonym_weight)
        return terms

    def rank(self, text: str, *, exclude: set[str] | None = None) -> list[Ranked]:
        terms = self.query_terms(text)
        excluded = exclude or set()
        ranked = []
        for name, counts in self.docs.items():
            if name in excluded:
                continue
            length_norm = self.k1 * (1 - self.b + self.b * self.lengths[name] / self.average)
            matched = {}
            for term, weight in terms.items():
                tf = counts.get(term)
                if not tf:
                    continue
                matched[term] = weight * self.idf[term] * tf * (self.k1 + 1) / (tf + length_norm)
            if matched:
                ranked.append(Ranked(name, sum(matched.values()), matched))
        return sorted(ranked, key=lambda item: (-item.score, item.name))

    def search(
        self,
        text: str,
        *,
        k: int = 6,
        always: tuple[str, ...] = (),
        exclude: set[str] | None = None,
    ) -> list[Ranked]:
        """Top-k tools for ``text``, plus ``always`` tools that exist in the catalogue."""
        ranked = self.rank(text, exclude=exclude)[:k]
        present = {item.name for item in ranked}
        for name in always:
            if name in self.catalogue and name not in present and name not in (exclude or set()):
                ranked.append(Ranked(name, 0.0, {"(always offered)": 0.0}))
        return ranked


def recall_at_k(retriever: ToolRetriever, cases: list[tuple[str, set[str]]], k: int) -> dict[str, Any]:
    """Share of cases whose every target tool is in the top-k, and per-tool hit rate."""
    full_hits = 0
    tool_hits = tool_total = 0
    misses = []
    for text, targets in cases:
        found = {item.name for item in retriever.search(text, k=k)}
        hit = targets <= found
        full_hits += hit
        tool_hits += len(targets & found)
        tool_total += len(targets)
        if not hit:
            misses.append({"request": text[:200], "missing": sorted(targets - found)})
    return {
        "k": k,
        "cases": len(cases),
        "all_targets_retrieved": full_hits,
        "target_tools": tool_total,
        "target_tools_retrieved": tool_hits,
        "misses": misses[:50],
    }
