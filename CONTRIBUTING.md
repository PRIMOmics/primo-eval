# Contributing

`primomics` computes the scores of the PRIMO benchmark. Scores must stay comparable
over time, so a PR that changes a score follows stricter rules than other PRs.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run the same checks as the CI:

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run --no-project --python 3.13 --with-requirements space/requirements.txt \
  --with gradio==5.50.0 --with pytest pytest space_tests
```

The last command tests the leaderboard Space, in `space/`, with the packages the Space
runs.

## Workflow

- `main` is the only long-lived branch. It always passes the CI.
- Create a short-lived branch from `main` for each change. If you cannot push to this
  repo, work from a fork.
- One pull request = one purpose. Prefer several small PRs to one large PR.
- Every code change comes with a test.
- A PR needs a review and a green CI before it is merged.
- PRs are merged with a merge commit, then their branch is deleted.
- Maintainers can split a large change into stacked PRs: each PR targets the branch
  of the previous one and says "Stacked on #N". Merge them in order. After each
  merge, GitHub moves the next PR onto `main`.
- Commit and PR titles are short sentences that start with a verb, for example
  "Add the linear probe". Write PR descriptions in plain, simple English: what changed
  and why.

## PRs that change a score

A score must never change by accident.

- Before you change a score, open an issue to discuss it. This covers the probe, the
  folds, a metric, the normalization, the data version, and any fix that changes a
  score.
- Say in the PR description whether scores change.
- Some tests check exact reference scores. Change these values only on purpose, and
  explain why in the PR.

## Code

- Python 3.12 or later. Type hints on every function in `src/`, in modern syntax
  (`list[str]`, `X | None`).
- Ruff for lint and format, with lines up to 88 characters.
- Keep it simple: write what the change needs, not code that "might be useful later".
- Small functions with clear names.
- No code comments: clear names and short docstrings explain the code. Docstrings
  are one line by default. Tool directives such as `# noqa` are fine.
- In the package, raise an exception instead of using `assert`, and use `logging`
  instead of `print`.
- Raise `SubmissionError` when the submitter has to fix something, and
  `EvaluatorError` when the problem is on our side.

## Tests

- Use pytest with small made-up data. Never use real labels or patient data.
- Made-up tasks and datasets get ids that start with 9 (`t901`, `d901`), so they
  never match a real one.

## Versions

A release is a git tag on `main` (for example `v0.2.0`) with a GitHub Release.
Until version 1.0.0, the version number says whether scores can change:

- the second number goes up when a score can change: `0.2.0` → `0.3.0`;
- the third number goes up otherwise (a fix that changes no score, docs, speed):
  `0.2.0` → `0.2.1`.

For stable scores, install a released version, not `main`.

Release steps:

1. Run `uv version 0.2.0`: the version goes from `0.2.0.dev0` to `0.2.0` in
   `pyproject.toml` and `uv.lock`. Then open a PR.
2. After the merge, tag the merge commit `v0.2.0` and create a GitHub Release. Its
   notes start with "Scores change" or "Scores do not change".
3. In the next PR, run `uv version 0.2.1.dev0` (the next patch with `.dev0`), so code
   taken from `main` never claims to be a release.
