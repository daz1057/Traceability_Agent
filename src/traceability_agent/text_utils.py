"""Utility helpers for deterministic natural-language heuristics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "by",
    "at",
    "from",
    "that",
    "this",
    "it",
    "is",
    "are",
    "be",
    "as",
    "so",
    "we",
    "i",
    "can",
    "will",
    "our",
    "their",
    "so",
    "not",
    "into",
    "about",
    "when",
    "then",
    "if",
    "but",
    "because",
    "due",
    "has",
    "have",
    "had",
}


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']+")


def normalise_text(text: str) -> str:
    """Return a lowercased, whitespace-collapsed version of ``text``."""

    return re.sub(r"\s+", " ", text.strip().lower())


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter used for heuristics."""

    fragments = re.split(r"[.!?]+", text)
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def keyword_set(text: str) -> Set[str]:
    """Return a keyword set excluding stopwords."""

    return {token.lower() for token in TOKEN_PATTERN.findall(text) if token.lower() not in STOPWORDS}


def keyphrase_candidates(text: str, min_len: int = 3, max_terms: int = 7) -> List[str]:
    """Extract simple keyphrases by filtering tokens and preserving order."""

    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    filtered: List[str] = []
    for token in tokens:
        if token in STOPWORDS or len(token) < min_len:
            continue
        filtered.append(token)
        if len(filtered) == max_terms:
            break
    return filtered


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Compute Jaccard similarity between two token sequences."""

    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cosine_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    """Simple cosine-like overlap using binary term frequency vectors."""

    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / math.sqrt(len(set_a) * len(set_b))


@dataclass(slots=True)
class PhraseExtraction:
    """Result of extracting persona/outcome/barrier phrases."""

    persona: str
    outcome: str
    barrier: str


def infer_persona(text: str, stakeholder: str | None = None) -> str:
    """Infer a persona string from text and optional stakeholder field."""

    if stakeholder:
        return stakeholder.strip()
    persona_match = re.search(r"as an? ([^,]+)", text, flags=re.IGNORECASE)
    if persona_match:
        return persona_match.group(1).strip()
    role_match = re.search(r"for ([A-Za-z ]+?) users", text, flags=re.IGNORECASE)
    if role_match:
        return role_match.group(1).strip().title()
    return "Stakeholder"


def extract_outcome_and_barrier(text: str) -> PhraseExtraction:
    """Derive desired outcome and barrier clauses from problem text."""

    outcome = "desired outcome"
    barrier = "an unspecified barrier"

    outcome_match = re.search(r"to ([^.,;]+)", text, flags=re.IGNORECASE)
    if outcome_match:
        outcome = outcome_match.group(1).strip()

    so_that_match = re.search(r"so that ([^.,;]+)", text, flags=re.IGNORECASE)
    if so_that_match:
        outcome = so_that_match.group(1).strip()

    barrier_match = re.search(r"because(?: of)? ([^.,;]+)", text, flags=re.IGNORECASE)
    if barrier_match:
        barrier = barrier_match.group(1).strip()
    else:
        due_to_match = re.search(r"due to ([^.,;]+)", text, flags=re.IGNORECASE)
        if due_to_match:
            barrier = due_to_match.group(1).strip()

    return PhraseExtraction(persona="", outcome=outcome, barrier=barrier)


def infer_value_intent(text: str) -> str:
    """Extract a value/intent clause from the text."""

    match = re.search(r"so that ([^.,;]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"in order to ([^.,;]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    sentences = split_sentences(text)
    if sentences:
        return sentences[-1]
    return text.strip()


def evidence_strength(text: str, stakeholder: str | None = None) -> int:
    """Infer an evidence strength score based on heuristics."""

    lowered = text.lower()
    if any(keyword in lowered for keyword in ("must", "required", "blocked", "regulatory")):
        return 2
    if stakeholder:
        return 2
    if any(keyword in lowered for keyword in ("need", "should", "struggle", "difficult")):
        return 1
    return 0
