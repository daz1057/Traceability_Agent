"""Problem normalisation logic."""

from __future__ import annotations

import re
from typing import List

from .data_models import NormalisedProblem, RawProblem, iter_domain_terms
from .text_utils import (
    evidence_strength,
    extract_outcome_and_barrier,
    infer_persona,
    infer_value_intent,
    keyphrase_candidates,
)


UTTERANCE_TYPE_RULES = [
    (re.compile(r"\bneed(s)?\b", re.IGNORECASE), "need"),
    (re.compile(r"\bwant\b", re.IGNORECASE), "need"),
    (re.compile(r"\brequest\b", re.IGNORECASE), "solution_request"),
    (re.compile(r"\bshould\b", re.IGNORECASE), "action_description"),
    (re.compile(r"\bcan't\b|cannot|unable", re.IGNORECASE), "failure"),
    (re.compile(r"\bfail|friction|pain|struggle\b", re.IGNORECASE), "pain_statement"),
]

DEFAULT_UTTERANCE = "pain_statement"


def classify_utterance(text: str) -> str:
    """Classify utterance type based on keyword heuristics."""

    for pattern, label in UTTERANCE_TYPE_RULES:
        if pattern.search(text):
            return label
    return DEFAULT_UTTERANCE


def normalise_problem(problem: RawProblem) -> NormalisedProblem:
    """Convert a raw problem into the canonical representation."""

    persona = infer_persona(problem.text, problem.stakeholder)
    phrases = extract_outcome_and_barrier(problem.text)
    value = infer_value_intent(problem.text)
    evidence = evidence_strength(problem.text, problem.stakeholder)
    utterance = classify_utterance(problem.text)
    domain_terms = iter_domain_terms(keyphrase_candidates(problem.text))

    desired_outcome = phrases.outcome
    barrier = phrases.barrier

    return NormalisedProblem(
        problem_id=problem.problem_id,
        raw_text=problem.text,
        utterance_type=utterance,
        persona=persona,
        desired_outcome=desired_outcome,
        barrier=barrier,
        value_intent=value,
        evidence_strength=evidence,
        domain_terms=domain_terms,
        stakeholder=problem.stakeholder,
        theme=problem.theme,
        metadata=problem.metadata,
    )


def normalise_problems(problems: List[RawProblem]) -> List[NormalisedProblem]:
    """Normalise a collection of problems."""

    return [normalise_problem(problem) for problem in problems]
