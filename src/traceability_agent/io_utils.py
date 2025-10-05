"""Input/output helpers for the traceability agent."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable, List

from .data_models import NormalisedProblem, ParsedStory, RawProblem, RawStory
from .normalisation import normalise_problems
from .pairing import AgentConfig, coverage_summaries, score_pairs
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
        "expression_type",
        "canonical_problem",
        "persona",
        "desired_outcome",
        "barrier",
        "domain_terms",
        "value_intent",
        "evidence_strength",
        "raw_text",
        "stakeholder",
        "theme",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for problem in problems:
            writer.writerow(
                {
                    "problem_id": problem.problem_id,
                    "expression_type": problem.expression_type,
                    "canonical_problem": problem.canonical_problem,
                    "persona": problem.persona,
                    "desired_outcome": problem.desired_outcome,
                    "barrier": problem.barrier,
                    "domain_terms": ", ".join(problem.domain_terms),
                    "value_intent": problem.value_intent,
                    "evidence_strength": problem.evidence_strength,
                    "raw_text": problem.raw_text,
                    "stakeholder": problem.stakeholder or "",
                    "theme": problem.theme or "",
                }
            )


def write_stories(path: Path, stories: List[ParsedStory]) -> None:
    fieldnames = [
        "story_id",
        "persona",
        "action_capability",
        "outcome",
        "value_intent",
        "domain_terms",
        "governance_signal",
        "raw_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for story in stories:
            writer.writerow(
                {
                    "story_id": story.story_id,
                    "persona": story.persona,
                    "action_capability": story.action_capability,
                    "outcome": story.outcome,
                    "value_intent": story.value_intent,
                    "domain_terms": ", ".join(story.domain_terms),
                    "governance_signal": story.governance_signal,
                    "raw_text": story.raw_text,
                }
            )


def write_edges(path: Path, edges: List[EdgeScore]) -> None:
    fieldnames = [
        "problem_id",
        "story_id",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "total_score",
        "confidence_band",
        "coverage_label",
        "facet_flags_json",
        "residual_coverage",
        "causal_rationale",
        "timestamp",
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
                    "D1": edge.d_scores["D1"],
                    "D2": edge.d_scores["D2"],
                    "D3": edge.d_scores["D3"],
                    "D4": edge.d_scores["D4"],
                    "D5": edge.d_scores["D5"],
                    "D6": edge.d_scores["D6"],
                    "D7": edge.d_scores["D7"],
                    "total_score": edge.total_score,
                    "confidence_band": edge.confidence_band,
                    "coverage_label": edge.coverage_label,
                    "facet_flags_json": json.dumps(edge.facet_flags, sort_keys=True),
                    "residual_coverage": edge.residual_coverage,
                    "causal_rationale": edge.causal_rationale,
                    "timestamp": edge.timestamp.isoformat(),
                    "flags": ", ".join(edge.flags),
                }
            )


def write_coverage(path: Path, summaries: List[CoverageSummary]) -> None:
    fieldnames = [
        "problem_id",
        "num_edges_high",
        "num_edges_medium",
        "residual_coverage_level",
        "unresolved_facets",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "problem_id": summary.problem_id,
                    "num_edges_high": summary.num_edges_high,
                    "num_edges_medium": summary.num_edges_medium,
                    "residual_coverage_level": summary.residual_coverage_level,
                    "unresolved_facets": ", ".join(summary.unresolved_facets),
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
    edges = score_pairs(normalised_problems, parsed_stories, config)
    summaries = coverage_summaries(edges, config.threshold)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_problems(output_path / "Problems_Normalised.csv", normalised_problems)
    write_stories(output_path / "Stories_Parsed.csv", parsed_stories)
    write_edges(output_path / "Edges.csv", edges)
    write_coverage(output_path / "Coverage_Summary.csv", summaries)
