"""Candidate pairing and scoring heuristics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

from .data_models import CoverageSummary, EdgeScore, NormalisedProblem, ParsedStory
from .text_utils import cosine_overlap, jaccard_similarity, keyword_set


@dataclass(slots=True)
class WeightConfig:
    """Scoring weights for the seven dimensions."""

    persona: float = 1.0
    capability: float = 2.0
    causal: float = 3.0
    granularity: float = 1.0
    value: float = 2.0
    governance: float = 1.0
    evidence: float = 1.0


@dataclass(slots=True)
class ThresholdConfig:
    """Thresholds used during scoring and coverage labelling."""

    high_confidence: int = 10
    medium_confidence: int = 6
    borderline_band: Tuple[int, int] = (8, 10)


@dataclass(slots=True)
class AgentConfig:
    """Configuration bundle for the agent."""

    weight: WeightConfig = field(default_factory=WeightConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    governance_terms: Sequence[str] = (
        "policy",
        "governance",
        "audit",
        "lineage",
        "approval",
    )


def persona_alignment(problem: NormalisedProblem, story: ParsedStory) -> int:
    prob_persona = problem.persona.lower()
    story_persona = story.persona.lower()
    if prob_persona == story_persona:
        return 2
    if prob_persona in story_persona or story_persona in prob_persona:
        return 1
    return 0


def capability_alignment(problem: NormalisedProblem, story: ParsedStory) -> int:
    problem_terms = keyword_set(problem.desired_outcome + " " + problem.barrier)
    story_terms = keyword_set(story.action_capability)
    if not story_terms:
        story_terms = keyword_set(story.raw_text)
    overlap = cosine_overlap(problem_terms, story_terms)
    if overlap >= 0.5:
        return 2
    if overlap >= 0.2:
        return 1
    return 0


def causal_coverage(problem: NormalisedProblem, story: ParsedStory) -> int:
    barrier_terms = keyword_set(problem.barrier)
    capability_terms = keyword_set(story.action_capability)
    if not barrier_terms or not capability_terms:
        return 0
    overlap = jaccard_similarity(barrier_terms, capability_terms)
    if overlap >= 0.4:
        return 2
    if overlap >= 0.2:
        return 1
    return 0


def granularity_fit(problem: NormalisedProblem, story: ParsedStory) -> int:
    problem_length = len(problem.desired_outcome.split()) + len(problem.barrier.split())
    story_length = len(story.action_capability.split())
    if story_length == 0:
        return 0
    ratio = problem_length / story_length
    if 0.5 <= ratio <= 1.5:
        return 2
    if 0.3 <= ratio <= 2.0:
        return 1
    return 0


def value_alignment(problem: NormalisedProblem, story: ParsedStory) -> int:
    problem_terms = keyword_set(problem.value_intent)
    story_terms = keyword_set(story.value_intent)
    overlap = jaccard_similarity(problem_terms, story_terms)
    if overlap >= 0.4:
        return 2
    if overlap >= 0.2:
        return 1
    return 0


def governance_alignment(problem: NormalisedProblem, story: ParsedStory, config: AgentConfig) -> int:
    problem_text = f"{problem.barrier} {problem.value_intent}"
    prob_terms = keyword_set(problem_text)
    story_terms = keyword_set(story.raw_text)
    strong_terms = {term for term in config.governance_terms}
    if story.governance_signal == 2 or any(term in prob_terms for term in strong_terms):
        if story.governance_signal >= 1 or any(term in story_terms for term in strong_terms):
            return 2
    if story.governance_signal >= 1 or (prob_terms & story_terms):
        return 1
    return 0


def evidence_transfer(problem: NormalisedProblem) -> int:
    return max(0, min(problem.evidence_strength, 2))


FACETS = (
    "covers_persona",
    "covers_capability",
    "covers_causal_root",
    "covers_value",
    "covers_governance",
    "covers_granularity",
)


def compute_facet_flags(d_scores: Dict[str, int]) -> Dict[str, bool]:
    return {
        "covers_persona": d_scores["D1"] == 2,
        "covers_capability": d_scores["D2"] >= 1,
        "covers_causal_root": d_scores["D3"] == 2,
        "covers_value": d_scores["D5"] >= 1,
        "covers_governance": d_scores["D6"] >= 1,
        "covers_granularity": d_scores["D4"] >= 1,
    }


def residual_coverage_level(facets: Dict[str, bool]) -> int:
    capability = facets["covers_capability"]
    causal_root = facets["covers_causal_root"]
    value = facets["covers_value"]
    coverage = sum([capability, causal_root, value])
    if coverage == 3:
        return 2
    if coverage >= 2:
        return 1
    return 0


def confidence_band(total_score: int, threshold: ThresholdConfig) -> str:
    if total_score >= threshold.high_confidence:
        return "High"
    if total_score >= threshold.medium_confidence:
        return "Medium"
    if total_score > 0:
        return "Low"
    return "None"


def coverage_label(d_scores: Dict[str, int], total: int, facets: Dict[str, bool], threshold: ThresholdConfig) -> str:
    residual = residual_coverage_level(facets)
    if (
        total >= threshold.high_confidence
        and residual == 2
        and d_scores["D3"] == 2
    ):
        return "Full"
    if total >= threshold.medium_confidence:
        return "Partial"
    return "None"


def causal_rationale(problem: NormalisedProblem, story: ParsedStory, d_scores: Dict[str, int]) -> str:
    if d_scores["D3"] == 2:
        return (
            f"{story.action_capability} removes the barrier '{problem.barrier}' so {problem.persona} achieves {problem.desired_outcome}."
        )
    if d_scores["D2"] >= 1:
        return (
            f"{story.action_capability} supports {problem.persona} towards {problem.desired_outcome} but does not fully remove '{problem.barrier}'."
        )
    return (
        f"{story.action_capability} is not clearly linked to overcoming '{problem.barrier}'."
    )


def candidate_pair(problem: NormalisedProblem, story: ParsedStory, config: AgentConfig) -> bool:
    persona_match = persona_alignment(problem, story) > 0
    domain_overlap = bool(set(problem.domain_terms) & set(story.domain_terms))
    governance_pair = (
        story.governance_signal >= 1
        and any(term in problem.barrier.lower() for term in config.governance_terms)
    )
    return persona_match or domain_overlap or governance_pair


def score_pair(problem: NormalisedProblem, story: ParsedStory, config: AgentConfig) -> EdgeScore:
    d_scores = {
        "D1": persona_alignment(problem, story),
        "D2": capability_alignment(problem, story),
        "D3": causal_coverage(problem, story),
        "D4": granularity_fit(problem, story),
        "D5": value_alignment(problem, story),
        "D6": governance_alignment(problem, story, config),
        "D7": evidence_transfer(problem),
    }

    total = sum(d_scores.values())
    facets = compute_facet_flags(d_scores)
    confidence = confidence_band(total, config.threshold)
    coverage = coverage_label(d_scores, total, facets, config.threshold)
    residual = residual_coverage_level(facets)
    rationale = causal_rationale(problem, story, d_scores)

    flags: List[str] = []
    low, high = config.threshold.borderline_band
    if low <= total <= high:
        flags.append("borderline")
    if confidence == "High" and d_scores["D7"] == 0:
        flags.append("low_evidence_high_confidence")

    return EdgeScore(
        problem_id=problem.problem_id,
        story_id=story.story_id,
        d_scores=d_scores,
        total_score=total,
        confidence_band=confidence,
        coverage_label=coverage,
        facet_flags=facets,
        residual_coverage=residual,
        causal_rationale=rationale,
        timestamp=datetime.utcnow(),
        flags=flags,
    )


def score_pairs(problems: List[NormalisedProblem], stories: List[ParsedStory], config: AgentConfig | None = None) -> List[EdgeScore]:
    if config is None:
        config = AgentConfig()
    edges: List[EdgeScore] = []
    for problem in problems:
        for story in stories:
            if candidate_pair(problem, story, config):
                edges.append(score_pair(problem, story, config))
    return edges


def coverage_summaries(edges: Iterable[EdgeScore], threshold: ThresholdConfig | None = None) -> List[CoverageSummary]:
    if threshold is None:
        threshold = ThresholdConfig()
    grouped: Dict[str, List[EdgeScore]] = defaultdict(list)
    for edge in edges:
        grouped[edge.problem_id].append(edge)

    summaries: List[CoverageSummary] = []
    for problem_id, entries in grouped.items():
        num_high = sum(1 for edge in entries if edge.confidence_band == "High")
        num_medium = sum(1 for edge in entries if edge.confidence_band == "Medium")
        best_residual = max((edge.residual_coverage for edge in entries), default=0)
        unresolved = [
            facet.replace("covers_", "")
            for facet in FACETS
            if not any(edge.facet_flags[facet] for edge in entries)
        ]
        summaries.append(
            CoverageSummary(
                problem_id=problem_id,
                num_edges_high=num_high,
                num_edges_medium=num_medium,
                residual_coverage_level=best_residual,
                unresolved_facets=unresolved,
            )
        )
    return summaries
