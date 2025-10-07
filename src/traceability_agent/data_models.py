"""Dataclasses describing the core entities handled by the traceability agent."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    raw_text: str
    utterance_type: str
    persona: str
    desired_outcome: str
    barrier: str
    value_intent: str
    domain_terms: List[str]
    evidence_strength: int
    stakeholder: Optional[str] = None
    theme: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def canonical_statement(self) -> str:
        """Return the canonical intent statement."""

        return f"{self.persona} cannot achieve {self.desired_outcome} because of {self.barrier}."


@dataclass(slots=True)
class ParsedStory:
    """User story parsed into comparable facets."""

    story_id: str
    raw_text: str
    persona: str
    capability: str
    functional_outcome: str
    business_value: str
    domain_terms: List[str]
    governance_signal: int
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredEdge:
    """Alignment score between a problem and a story."""

    problem_id: str
    story_id: str
    scores: Dict[str, int]
    total_score: int
    confidence_band: str
    facet_coverage: Dict[str, bool]
    coverage_label: str
    causal_rationale: str
    provenance: Dict[str, object]
    flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class CoverageSummary:
    """Summary of coverage and review decisions for a given problem."""

    problem_id: str
    best_confidence: str
    best_coverage: str
    unresolved_facets: List[str]
    escalate: bool
    escalate_reasons: List[str]


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
