"""Input/output helpers for the traceability agent."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable, List

from .data_models import NormalisedProblem, ParsedStory, RawProblem, RawStory, ScoredEdge
from .normalisation import normalise_problems
from .pairing import AgentConfig, coverage_summaries, propose_pairs, score_pairs
from .story_parser import parse_stories


PROBLEM_FIELDNAMES = ["problem_id", "text", "stakeholder", "theme"]
STORY_FIELDNAMES = ["story_id", "text", "rationale"]


def read_json_lines(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_problems(path: str | os.PathLike[str]) -> List[RawProblem]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    problems: List[RawProblem] = []
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                problems.append(
                    RawProblem(
                        problem_id=row.get("problem_id") or row.get("PR_ID") or row.get("id") or str(len(problems) + 1),
                        text=row.get("text") or row.get("problem_text") or "",
                        stakeholder=row.get("stakeholder") or row.get("persona"),
                        theme=row.get("theme"),
                        metadata={key: value for key, value in row.items() if key not in PROBLEM_FIELDNAMES},
                    )
                )
    elif file_path.suffix.lower() in {".json", ".jsonl"}:
        records = read_json_lines(file_path) if file_path.suffix.lower() == ".jsonl" else json.loads(file_path.read_text("utf-8"))
        for record in records:
            problems.append(
                RawProblem(
                    problem_id=str(record.get("problem_id") or record.get("id")),
                    text=record.get("text") or record.get("problem_text") or "",
                    stakeholder=record.get("stakeholder"),
                    theme=record.get("theme"),
                    metadata={key: value for key, value in record.items() if key not in PROBLEM_FIELDNAMES},
                )
            )
    else:
        raise ValueError(f"Unsupported problem file format: {file_path.suffix}")
    return problems


STORY_HEADING_RE = re.compile(r"^#### (?P<id>[A-Za-z0-9\-]+): (?P<title>.+)$")
STORY_BULLET_RE = re.compile(r"^- As an? .+$", re.IGNORECASE)


def parse_story_blocks(lines: Iterable[str]) -> List[RawStory]:
    stories: List[RawStory] = []
    current_id: str | None = None
    current_text: List[str] = []
    current_rationale: List[str] = []
    in_story = False
    for line in lines:
        stripped = line.strip()
        heading = STORY_HEADING_RE.match(stripped)
        if heading:
            if current_id and current_text:
                stories.append(
                    RawStory(
                        story_id=current_id,
                        text=" ".join(current_text).strip(),
                        rationale="\n".join(current_rationale).strip() or None,
                    )
                )
            current_id = heading.group("id")
            current_text = []
            current_rationale = []
            in_story = False
            continue
        if STORY_BULLET_RE.match(stripped):
            in_story = True
            current_text.append(stripped.lstrip("- "))
            continue
        if stripped.startswith("- ") and in_story:
            current_text.append(stripped.lstrip("- "))
            continue
        if stripped.startswith("- Acceptance Criteria"):
            in_story = False
            continue
        if stripped.startswith("  - "):
            current_rationale.append(stripped.lstrip("- "))
    if current_id and current_text:
        stories.append(
            RawStory(
                story_id=current_id,
                text=" ".join(current_text).strip(),
                rationale="\n".join(current_rationale).strip() or None,
            )
        )
    return stories


def load_stories(path: str | os.PathLike[str]) -> List[RawStory]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() in {".md", ".txt"}:
        lines = file_path.read_text("utf-8").splitlines()
        return parse_story_blocks(lines)
    if file_path.suffix.lower() == ".csv":
        stories: List[RawStory] = []
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stories.append(
                    RawStory(
                        story_id=row.get("story_id") or row.get("BR_ID") or row.get("id") or str(len(stories) + 1),
                        text=row.get("text") or row.get("story_text") or "",
                        rationale=row.get("rationale"),
                        metadata={key: value for key, value in row.items() if key not in STORY_FIELDNAMES},
                    )
                )
        return stories
    if file_path.suffix.lower() in {".json", ".jsonl"}:
        records = read_json_lines(file_path) if file_path.suffix.lower() == ".jsonl" else json.loads(file_path.read_text("utf-8"))
        stories = []
        for record in records:
            stories.append(
                RawStory(
                    story_id=str(record.get("story_id") or record.get("id")),
                    text=record.get("text") or record.get("story") or "",
                    rationale=record.get("rationale"),
                    metadata={key: value for key, value in record.items() if key not in STORY_FIELDNAMES},
                )
            )
        return stories
    raise ValueError(f"Unsupported story file format: {file_path.suffix}")


def write_problems(path: Path, problems: List[NormalisedProblem]) -> None:
    fieldnames = [
        "problem_id",
        "raw_text",
        "utterance_type",
        "persona",
        "desired_outcome",
        "barrier",
        "value_intent",
        "domain_terms",
        "evidence_strength",
        "stakeholder",
        "theme",
        "canonical_statement",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for problem in problems:
            writer.writerow(
                {
                    "problem_id": problem.problem_id,
                    "raw_text": problem.raw_text,
                    "utterance_type": problem.utterance_type,
                    "persona": problem.persona,
                    "desired_outcome": problem.desired_outcome,
                    "barrier": problem.barrier,
                    "value_intent": problem.value_intent,
                    "domain_terms": ", ".join(problem.domain_terms),
                    "evidence_strength": problem.evidence_strength,
                    "stakeholder": problem.stakeholder or "",
                    "theme": problem.theme or "",
                    "canonical_statement": problem.canonical_statement,
                }
            )


def write_stories(path: Path, stories: List[ParsedStory]) -> None:
    fieldnames = [
        "story_id",
        "raw_text",
        "persona",
        "capability",
        "functional_outcome",
        "business_value",
        "domain_terms",
        "governance_signal",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for story in stories:
            writer.writerow(
                {
                    "story_id": story.story_id,
                    "raw_text": story.raw_text,
                    "persona": story.persona,
                    "capability": story.capability,
                    "functional_outcome": story.functional_outcome,
                    "business_value": story.business_value,
                    "domain_terms": ", ".join(story.domain_terms),
                    "governance_signal": story.governance_signal,
                }
            )


def write_edges(path: Path, edges: List[ScoredEdge]) -> None:
    fieldnames = [
        "problem_id",
        "story_id",
        "persona_alignment",
        "capability_alignment",
        "causal_coverage",
        "granularity_fit",
        "value_alignment",
        "governance_alignment",
        "evidence_strength_transfer",
        "total_score",
        "confidence_band",
        "coverage_label",
        "facet_coverage_json",
        "causal_rationale",
        "provenance_json",
        "flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "problem_id": edge.problem_id,
                    "story_id": edge.story_id,
                    "persona_alignment": edge.scores["persona_alignment"],
                    "capability_alignment": edge.scores["capability_alignment"],
                    "causal_coverage": edge.scores["causal_coverage"],
                    "granularity_fit": edge.scores["granularity_fit"],
                    "value_alignment": edge.scores["value_alignment"],
                    "governance_alignment": edge.scores["governance_alignment"],
                    "evidence_strength_transfer": edge.scores["evidence_strength_transfer"],
                    "total_score": edge.total_score,
                    "confidence_band": edge.confidence_band,
                    "coverage_label": edge.coverage_label,
                    "facet_coverage_json": json.dumps(edge.facet_coverage, sort_keys=True),
                    "causal_rationale": edge.causal_rationale,
                    "provenance_json": json.dumps(edge.provenance, sort_keys=True),
                    "flags": ", ".join(edge.flags),
                }
            )


def write_coverage(path: Path, summaries: List[CoverageSummary]) -> None:
    fieldnames = [
        "problem_id",
        "best_confidence",
        "best_coverage",
        "unresolved_facets",
        "escalate",
        "escalate_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "problem_id": summary.problem_id,
                    "unresolved_facets": ", ".join(summary.unresolved_facets),
                    "best_confidence": summary.best_confidence,
                    "best_coverage": summary.best_coverage,
                    "escalate": "yes" if summary.escalate else "no",
                    "escalate_reasons": ", ".join(summary.escalate_reasons),
                }
            )


def run_pipeline(
    problems_path: str | os.PathLike[str],
    stories_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: AgentConfig | None = None,
) -> None:
    if config is None:
        config = AgentConfig()
    problems = load_problems(problems_path)
    stories = load_stories(stories_path)

    normalised_problems = normalise_problems(problems)
    parsed_stories = parse_stories(stories)
    candidate_pairs = propose_pairs(normalised_problems, parsed_stories, config)
    edges = score_pairs(candidate_pairs, config)
    summaries = coverage_summaries(normalised_problems, edges)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_problems(output_path / "Problems_Normalised.csv", normalised_problems)
    write_stories(output_path / "Stories_Parsed.csv", parsed_stories)
    write_edges(output_path / "Edges.csv", edges)
    write_coverage(output_path / "Coverage_Summary.csv", summaries)
