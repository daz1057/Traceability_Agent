"""Candidate pairing, scoring, and review heuristics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

from .data_models import CoverageSummary, NormalisedProblem, ParsedStory, ScoredEdge
from .text_utils import cosine_overlap, jaccard_similarity, keyword_set, normalise_text


ESSENTIAL_FACETS = ("capability", "causal_root", "value")


@dataclass(slots=True)
class ThresholdConfig:
    """Thresholds used during scoring and coverage labelling."""

    high_confidence: int = 12
    medium_confidence: int = 8
    borderline_band: Tuple[int, int] = (8, 11)


@dataclass(slots=True)
class AgentConfig:
    """Configuration bundle for the agent."""

    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    governance_terms: Sequence[str] = (
        "policy",
        "governance",
        "audit",
        "lineage",
        "approval",
        "control",
        "compliance",
    )


def _role_tokens(role: str) -> set[str]:
    return {token for token in normalise_text(role).split() if token}


def persona_alignment(problem: NormalisedProblem, story: ParsedStory) -> int:
    """Score persona alignment (0–2)."""

    prob_tokens = _role_tokens(problem.persona)
    story_tokens = _role_tokens(story.persona)
    if not prob_tokens or not story_tokens:
        return 0
    if prob_tokens == story_tokens or prob_tokens.issubset(story_tokens) or story_tokens.issubset(prob_tokens):
        return 2
    if prob_tokens & story_tokens:
        return 1
    return 0


def capability_alignment(problem: NormalisedProblem, story: ParsedStory) -> int:
    """Score capability alignment (0–2)."""

    problem_terms = keyword_set(f"{problem.desired_outcome} {problem.barrier}")
    story_terms = keyword_set(story.capability)
    if not story_terms:
        story_terms = keyword_set(story.raw_text)
    overlap = cosine_overlap(problem_terms, story_terms)
    if overlap >= 0.5:
        return 2
    if overlap >= 0.2:
        return 1
    return 0


def causal_coverage(problem: NormalisedProblem, story: ParsedStory) -> int:
    """Score causal coverage (0–2)."""

    barrier_terms = keyword_set(problem.barrier)
    capability_terms = keyword_set(story.capability)
    if not barrier_terms or not capability_terms:
        return 0
    overlap = jaccard_similarity(barrier_terms, capability_terms)
    if overlap >= 0.4:
        return 2
    if overlap >= 0.2:
        return 1
    return 0


def granularity_fit(problem: NormalisedProblem, story: ParsedStory) -> int:
    """Score granularity fit (0–2)."""

    problem_length = len(problem.desired_outcome.split()) + len(problem.barrier.split())
    story_length = len(story.capability.split())
    if story_length == 0:
        return 0
    ratio = problem_length / story_length
    if 0.5 <= ratio <= 1.5:
        return 2
    if 0.3 <= ratio <= 2.0:
        return 1
    return 0


def value_alignment(problem: NormalisedProblem, story: ParsedStory) -> int:
    """Score value alignment (0–2)."""

    problem_terms = keyword_set(problem.value_intent)
    story_terms = keyword_set(story.business_value)
    overlap = jaccard_similarity(problem_terms, story_terms)
    if overlap >= 0.4:
        return 2
    if overlap >= 0.2:
        return 1
    return 0


def governance_alignment(problem: NormalisedProblem, story: ParsedStory, config: AgentConfig) -> int:
    """Score governance/policy alignment (0–2)."""

    problem_terms = keyword_set(f"{problem.barrier} {problem.value_intent}")
    story_terms = keyword_set(story.raw_text)
    strong_terms = set(config.governance_terms)
    if story.governance_signal == 2 or (problem_terms & strong_terms):
        if story.governance_signal >= 1 or (story_terms & strong_terms):
            return 2
    if story.governance_signal >= 1 or (problem_terms & story_terms):
        return 1
    return 0


def evidence_transfer(problem: NormalisedProblem) -> int:
    """Transfer evidence strength (0–2)."""

    return max(0, min(problem.evidence_strength, 2))


def compute_facet_flags(scores: Dict[str, int]) -> Dict[str, bool]:
    """Return facet coverage flags using rubric definitions."""

    return {
        "persona": scores["persona_alignment"] == 2,
        "capability": scores["capability_alignment"] >= 1,
        "causal_root": scores["causal_coverage"] == 2,
        "value": scores["value_alignment"] >= 1,
        "governance": scores["governance_alignment"] >= 1,
        "granularity_compatible": scores["granularity_fit"] >= 1,
    }


def confidence_band(total_score: int, problem: NormalisedProblem, scores: Dict[str, int], threshold: ThresholdConfig) -> str:
    """Derive confidence band with vision caps applied."""

    if total_score >= threshold.high_confidence:
        band = "High"
    elif total_score >= threshold.medium_confidence:
        band = "Medium"
    elif total_score > 0:
        band = "Low"
    else:
        band = "None"

    if problem.evidence_strength == 0 and band == "High":
        if not (scores["capability_alignment"] == 2 and scores["causal_coverage"] == 2):
            band = "Medium"
    return band


def coverage_label(band: str, facets: Dict[str, bool]) -> str:
    """Label coverage according to essentials-first rule."""

    if facets["capability"] and facets["causal_root"] and facets["value"]:
        return "Full"
    if band in {"High", "Medium"}:
        return "Partial"
    return "None"


def causal_rationale(problem: NormalisedProblem, story: ParsedStory, scores: Dict[str, int]) -> str:
    """Generate a single-sentence rationale."""

    if scores["causal_coverage"] == 2:
        return (
            f"{story.capability} neutralises the barrier '{problem.barrier}', enabling {problem.persona} to achieve {problem.desired_outcome}."
        )
    if scores["capability_alignment"] >= 1:
        return (
            f"{story.capability} helps {problem.persona} progress towards {problem.desired_outcome} but does not fully remove '{problem.barrier}'."
        )
    return (
        f"{story.capability} does not address the barrier '{problem.barrier}'."
    )


def candidate_pair(problem: NormalisedProblem, story: ParsedStory, config: AgentConfig) -> bool:
    """Determine whether a problem/story pair should be scored."""

    persona_match = persona_alignment(problem, story) > 0
    domain_overlap = bool(set(problem.domain_terms) & set(story.domain_terms))
    governance_bridge = (
        story.governance_signal >= 1
        and any(term in problem.barrier.lower() for term in config.governance_terms)
    )
    return persona_match or domain_overlap or governance_bridge


def propose_pairs(
    problems: List[NormalisedProblem], stories: List[ParsedStory], config: AgentConfig | None = None
) -> List[Tuple[NormalisedProblem, ParsedStory]]:
    """Stage 3 pairing – return candidate problem/story tuples."""

    if config is None:
        config = AgentConfig()
    pairs: List[Tuple[NormalisedProblem, ParsedStory]] = []
    for problem in problems:
        for story in stories:
            if candidate_pair(problem, story, config):
                pairs.append((problem, story))
    return pairs


def score_pair(problem: NormalisedProblem, story: ParsedStory, config: AgentConfig) -> ScoredEdge:
    """Score a single candidate pair."""

    scores = {
        "persona_alignment": persona_alignment(problem, story),
        "capability_alignment": capability_alignment(problem, story),
        "causal_coverage": causal_coverage(problem, story),
        "granularity_fit": granularity_fit(problem, story),
        "value_alignment": value_alignment(problem, story),
        "governance_alignment": governance_alignment(problem, story, config),
        "evidence_strength_transfer": evidence_transfer(problem),
    }
    total = sum(scores.values())
    facets = compute_facet_flags(scores)
    band = confidence_band(total, problem, scores, config.threshold)
    coverage = coverage_label(band, facets)
    rationale = causal_rationale(problem, story, scores)

    flags: List[str] = []
    low, high = config.threshold.borderline_band
    if low <= total <= high and band == "Medium":
        flags.append("borderline_medium")
    if band == "High" and problem.evidence_strength <= 1:
        flags.append("high_needs_review")

    provenance = {
        "created_at": datetime.utcnow().isoformat(),
        "prompt_versions": {
            "problem_norm": "v1",
            "story_parse": "v1",
            "causal_judge": "v1",
            "value_align": "v1",
        },
        "notes": "heuristic_rule_based",
    }

    return ScoredEdge(
        problem_id=problem.problem_id,
        story_id=story.story_id,
        scores=scores,
        total_score=total,
        confidence_band=band,
        facet_coverage=facets,
        coverage_label=coverage,
        causal_rationale=rationale,
        provenance=provenance,
        flags=flags,
    )


def score_pairs(
    pairs: Iterable[Tuple[NormalisedProblem, ParsedStory]], config: AgentConfig | None = None
) -> List[ScoredEdge]:
    """Stage 4 scoring – evaluate each candidate pair."""

    if config is None:
        config = AgentConfig()
    edges: List[ScoredEdge] = []
    for problem, story in pairs:
        edges.append(score_pair(problem, story, config))
    return edges


def coverage_summaries(
    problems: List[NormalisedProblem], edges: Iterable[ScoredEdge]
) -> List[CoverageSummary]:
    """Stage 5 review summaries and escalation hints."""

    problem_lookup: Dict[str, NormalisedProblem] = {problem.problem_id: problem for problem in problems}
    grouped: Dict[str, List[ScoredEdge]] = defaultdict(list)
    for edge in edges:
        grouped[edge.problem_id].append(edge)

    summaries: List[CoverageSummary] = []
    for problem_id, entries in grouped.items():
        best_edge = max(entries, key=lambda edge: edge.total_score)
        unresolved_facets = [
            facet
            for facet in ("persona", "capability", "causal_root", "value", "governance", "granularity_compatible")
            if not any(edge.facet_coverage[facet] for edge in entries)
        ]

        reasons: List[str] = []
        if not any(edge.coverage_label == "Full" for edge in entries):
            reasons.append("no_full_coverage")
        if any("borderline" in "_".join(edge.flags) for edge in entries):
            reasons.append("borderline_medium")
        problem = problem_lookup.get(problem_id)
        if problem and any(
            edge.confidence_band == "High" and problem.evidence_strength <= 1 for edge in entries
        ):
            reasons.append("high_with_low_evidence")
        if any(facet in unresolved_facets for facet in ESSENTIAL_FACETS):
            reasons.append("residual_gaps")

        summaries.append(
            CoverageSummary(
                problem_id=problem_id,
                best_confidence=best_edge.confidence_band,
                best_coverage=best_edge.coverage_label,
                unresolved_facets=unresolved_facets,
                escalate=bool(reasons),
                escalate_reasons=sorted(set(reasons)),
            )
        )

    # Include problems without any candidate edges for completeness.
    for problem in problems:
        if problem.problem_id not in grouped:
            summaries.append(
                CoverageSummary(
                    problem_id=problem.problem_id,
                    best_confidence="None",
                    best_coverage="None",
                    unresolved_facets=["persona", "capability", "causal_root", "value", "governance", "granularity_compatible"],
                    escalate=True,
                    escalate_reasons=["no_pairs"],
                )
            )

    return summaries
