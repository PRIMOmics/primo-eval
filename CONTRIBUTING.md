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
uv run --isolated --no-project --python 3.13 \
  --with-requirements space/requirements.txt \
  --with "gradio[oauth]==5.50.0" --with pytest \
  pytest space_tests
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
- A merge does not deploy the leaderboard Space: see "Deploy the Space" below.
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

### Board versions

The second number is also the version of the leaderboard board: the `v0.2.x`
releases all score board version `0.2`. A new board version opens when a score can
change, and a change of the task set counts: a task or a dataset added or retired.
A new model never opens a version. `space/versions.yaml` lists the board versions
and the Hub commits that each one reads.

A release that changes scores also opens a board version:

1. In the release PR, add the new version at the top of `space/versions.yaml`,
   with the commits of the data, the labels and the HVG features it reads, and set
   `current` to it.
2. After the merge, tag the release as above. Also tag the commits of step 1 `v0.2`
   on the three Hub dataset repos.
3. From the tag, score the live entries of the previous version again from their
   archived files, with the packages of the Space:

   ```bash
   HF_TOKEN=... uv run --isolated --no-project --python 3.13 \
     --with-requirements space/requirements.txt \
     python space/rescore.py --from 0.1 --checkpoint-dir ckpt --publish
   ```

   The Space still runs version `0.1`, which shows only its own rows. Add
   `--dry-run` first to list the entries without scoring them. If an entry fails,
   or gets no score on a task it held, nothing is published: fix it, or rerun with
   `--models` and the entries to keep.
4. Deploy the tag (see "Deploy the Space").
5. Run step 3 again: it adds the submissions made before the deploy.

Step 1 pins the commit of the new data, so the data is on the Hub before the
release PR. Deploy soon after it: until the deploy, the quickstart and
`primo download` already read the new data, and the live Space refuses a file with
a dataset that its version does not list.

### Open version 0.1

Version `0.1` is the first board version. The rows written before versions existed
carry none, so this version opens once, by hand:

1. In `task_results.csv`, set `version` to `0.1` on every row with a `submitted_at`
   of `2026-10-19 15:22:47` or later. These rows hold the scores of version `0.1`.
2. Run `rescore.py --from legacy --publish`, as in step 3 above. It scores again,
   into version `0.1`, the live entries that step 1 did not tag.
3. Deploy the release, then repeat step 1 for the rows written before the deploy.
4. Tag the commits of version `0.1` `v0.1` on the three Hub dataset repos.

## Deploy the Space

The leaderboard Space runs the `space/` folder of a release, so the rows of a board
version all come from the code of that version. After a release that changes
`space/`, a maintainer with write access to the `PRIMOmics/primo-eval` Space deploys it
by hand, with their own Hugging Face login (`uv run hf auth login`):

```bash
git fetch origin --tags
tag=v0.2.0
tmp=$(mktemp -d) && git archive "$tag" space | tar -x -C "$tmp"
uv run hf upload PRIMOmics/primo-eval "$tmp/space" . --repo-type space \
  --delete "*" --commit-message "Deploy $tag"
```

- `git archive` copies `space/` as it is in the release, without local changes or
  cache files.
- `--delete "*"` also deletes the Space files that are no longer in `space/`. The Hub
  always keeps `.gitattributes`.
- The commit message says which release the Space runs.
- If your account cannot write to the Space, the Hub refuses the upload.
