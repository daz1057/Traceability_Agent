"""Command line interface for running the traceability agent pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the traceability agent pipeline")
    parser.add_argument("problems", help="Path to problem statements (CSV/JSON/JSONL)")
    parser.add_argument("stories", help="Path to user stories (Markdown/CSV/JSON)")
    parser.add_argument("output", help="Output directory for generated artefacts")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args.problems, args.stories, args.output)
    output_path = Path(args.output)
    print(f"Artifacts written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
