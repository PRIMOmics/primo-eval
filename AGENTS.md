# AGENTS.md

Guidance for coding agents in this repository. Read `CONTRIBUTING.md` before any
change: its rules apply to every change.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check --fix
uv run ruff format
```

## Layout

- `src/primomics/submission.py`: reads and checks a submission file.
- `src/primomics/probe.py`: the fixed linear probe.
- `src/primomics/metrics.py`: AUROC, Pearson and sample-wise Spearman.
- `src/primomics/evaluate.py`: scores one task, then a whole submission.
- `src/primomics/boards.py` and `ranking.py`: the boards and the Elo ranking on each board.
- `src/primomics/tasks.py` and `public_tasks.yaml`: the specs of the public tasks and
  their reader.
- `src/primomics/data.py`: downloads the benchmark data from the Hub, records it in
  `primo-lock.json`, and checks a submission file against it.
- `src/primomics/cli.py`: the `primo` command.
- `src/primomics/errors.py`: `SubmissionError` (the submitter fixes it) and
  `EvaluatorError` (we fix it).
- `tests/`: one test file per module.

## Rules

- A score must never change by accident. Some tests check exact reference scores: if
  one fails, find the cause before you change the test.
- Do not add code comments.
- A hook runs `ruff format` and `ruff check --fix` when you finish.
