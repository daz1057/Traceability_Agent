# Traceability_Agent

Match the problems in a project to the user stories.

## Overview

This repository contains a lightweight, fully deterministic Python implementation of the traceability agent described in `Requirements/Notes.txt`. The pipeline performs four major steps:

1. **Problem normalisation** – raw problem statements are classified by expression type and rewritten into the canonical `"<Persona> cannot achieve <Desired Outcome> because of <Barrier>"` form while extracting supporting facets.
2. **Story parsing** – user stories are parsed into persona, capability, value intent, domain terms, and a governance signal so they can be compared conceptually to problems.
3. **Pair scoring** – candidate problem/story pairs are generated when personas, domains, or governance themes overlap, then scored across the seven decision dimensions (D1–D7) with confidence bands and coverage labels.
4. **Reporting** – normalised nodes, scored edges, and coverage summaries are exported to CSV artefacts that can be consumed by downstream traceability matrices.

All heuristics are rule-based to guarantee reproducibility and remain faithful to the operating specification.

## Installation

The project uses a standard `pyproject.toml` layout. Install the package in editable mode to experiment locally:

```bash
pip install -e .
```

Python 3.10 or newer is required.

## Usage

Run the end-to-end pipeline with the `traceability-agent` CLI. Supply the raw problem and story inputs (CSV/JSON/JSONL for problems, Markdown/CSV/JSON for stories) and the output directory where the generated artefacts should be written.

```bash
traceability-agent examples/problems.csv examples/stories.md out
```

The command produces the following CSV files:

- `Problems_Normalised.csv`
- `Stories_Parsed.csv`
- `Edges.csv`
- `Coverage_Summary.csv`

Each run is deterministic for the same inputs, enabling reproducible audits.

### Running inside VS Code

The project does not host a web server. If you open `http://localhost:8080` (or any other port) in a browser while the CLI is running you will see a “connection refused” page because nothing is listening there. To work with the project in VS Code:

1. Open the folder in VS Code (`File → Open Folder…`).
2. Create/activate a Python environment and install the package in editable mode with `pip install -e .`.
3. Use the integrated terminal to invoke the CLI command shown above. The results are written to the output directory, not served over HTTP, so inspect the generated CSV files directly from the filesystem.

## Examples

Sample inputs for quick experimentation are available in the `examples/` directory. The generated CSV outputs can be inspected to validate the heuristic behaviour and adapt the rules for new datasets.
