"""Dataclasses describing the core entities handled by the traceability agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional


@dataclass(slots=True)
class RawProblem:
    """Raw problem statement gathered from discovery artefacts."""

    problem_id: str
    text: str
    stakeholder: Optional[str] = None
    theme: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RawStory:
    """Raw user story text."""

    story_id: str
    text: str
    rationale: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class NormalisedProblem:
    """Problem converted into the canonical schema."""

    problem_id: str
    expression_type: str
    canonical_problem: str
    persona: str
    desired_outcome: str
    barrier: str
    domain_terms: List[str]
    value_intent: str
    evidence_strength: int
    raw_text: str
    stakeholder: Optional[str] = None
    theme: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedStory:
    """User story parsed into comparable facets."""

    story_id: str
    persona: str
    action_capability: str
    outcome: str
    value_intent: str
    domain_terms: List[str]
    governance_signal: int
    raw_text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EdgeScore:
    """Alignment score between a problem and a story."""

    problem_id: str
    story_id: str
    d_scores: Dict[str, int]
    total_score: int
    confidence_band: str
    coverage_label: str
    facet_flags: Dict[str, bool]
    residual_coverage: int
    causal_rationale: str
    timestamp: datetime
    flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class CoverageSummary:
    """Summary of coverage for a given problem."""

    problem_id: str
    num_edges_high: int
    num_edges_medium: int
    residual_coverage_level: int
    unresolved_facets: List[str]


def iter_domain_terms(terms: Iterable[str]) -> List[str]:
    """Return a de-duplicated list of domain terms preserving order."""

    seen = set()
    ordered: List[str] = []
    for term in terms:
        normalised = term.strip().lower()
        if not normalised:
            continue
        if normalised in seen:
            continue
        ordered.append(normalised)
        seen.add(normalised)
    return ordered
