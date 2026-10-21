# AGENTS.md

Guidance for coding agents in this repository. Read `CONTRIBUTING.md` before any
change: its rules apply to every change.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check --fix
uv run ruff format
uv run --isolated --no-project --python 3.13 \
  --with-requirements space/requirements.txt \
  --with "gradio[oauth]==5.50.0" --with pytest \
  pytest space_tests
```

The last command runs the Space tests with the packages of the Space: the Python
and Gradio versions of `space/README.md`, and `space/requirements.txt`. Gradio
comes with its `oauth` extra, which the login button of the Space needs.

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
- `src/primomics/submit.py`: sends a submission file to the leaderboard Space.
- `src/primomics/cli.py`: the `primo` command.
- `src/primomics/errors.py`: `SubmissionError` (the submitter fixes it) and
  `EvaluatorError` (we fix it).
- `tests/`: one test file per module.
- `space/`: the leaderboard Space, as it runs on Hugging Face. `space/README.md` holds
  its config. It still has its own scorer, `space/evaluator.py`. Maintainers deploy
  a release of it by hand (see `CONTRIBUTING.md`).
- `space/versions.yaml`: the board versions and the Hub commits each one reads.
  `space/rescore.py` fills the board of a new version from the archived files.
- `space_tests/`: the Space tests, one file per Space module.
- `scripts/baselines.py`: builds the linear baselines, such as `HVG-1200-genes`,
  from the public data; maintainers score and publish them with the scorer of
  `space/`.

## Rules

- A score must never change by accident. Some tests check exact reference scores: if
  one fails, find the cause before you change the test.
- Do not add code comments.
- `space/` imports only its own modules and the packages of `space/requirements.txt`,
  so Hugging Face can build it as is.
- A hook runs `ruff format` and `ruff check --fix` when you finish.
