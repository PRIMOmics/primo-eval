"""Score archived PRIMO submissions concatenated with the HVG-1200 features.

The Space scores every new submission twice: as sent, and concatenated with our
``HVG-1200-genes`` baseline features, published as ``<model> + HVG``. This
gives the same ``+ HVG`` entry to every archived submission still live on the
board. Rerun it after republishing the HVG baseline with ``scripts/baselines.py``,
so every ``+ HVG`` entry is scored against the current reference. It needs
``HF_TOKEN``, with read access to the labels and the results, and write access
to the results for ``--publish``:

    uv run python scripts/score_hvg_concat.py \\
        --checkpoint-dir checkpoints [--models NAME ...] [--publish]
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
if str(SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(SPACE_DIR))

from evaluator import (
    EMBEDDING,
    GENE_SPACE,
    RESULTS_REPO,
    TASK_ID,
    EvaluatorError,
    SubmissionError,
    fetch_tasks_registry,
    score_all,
)
from leaderboard import latest_only
from results import (
    HVG_TAG,
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    IS_BASELINE,
    MODEL_NAME,
    OWNER,
    PAPER_LINK,
    SUBMISSION_ID,
    append_results,
    fetch_hvg_reference,
    hvg_name,
    read_results,
    task_rows,
    with_owner,
)

logger = logging.getLogger(__name__)

ARCHIVE_PREFIX = "models/"
MANIFEST_FILENAME = "manifest.json"
GENE_SPACE_MODELS = ("COMPASS",)


@dataclass(frozen=True)
class LiveSubmission:
    """An archived submission whose rows still back a plain leaderboard entry."""

    manifest: dict
    credits: dict


def live_submissions(
    results: pd.DataFrame, manifests: list[dict], task_ids: set[str]
) -> list[LiveSubmission]:
    """Keep the archived submissions that still back a plain leaderboard entry.

    Baselines and existing ``+ HVG`` entries are skipped. A superseded
    submission is also skipped, because no latest row carries its id, and so is
    one whose latest rows are all on tasks retired from the registry.
    """
    latest = with_owner(latest_only(results))
    plain = latest[
        ~latest[IS_BASELINE]
        & ~latest[MODEL_NAME].astype(str).str.endswith(HVG_TAG)
        & latest[TASK_ID].astype(str).isin(task_ids)
    ]
    live = []
    for manifest in manifests:
        rows = plain[
            (plain[SUBMISSION_ID] == manifest["submission_id"])
            & (plain[MODEL_NAME] == manifest["model_name"])
        ]
        if not rows.empty:
            parent = rows.iloc[-1]
            credits = {
                key: parent[key]
                for key in (OWNER, INSTITUTION, IS_AUTHOR_SUBMISSION, PAPER_LINK)
            }
            live.append(LiveSubmission(manifest, credits))
    return live


def hvg_rows(submission: LiveSubmission, result: dict, submitted_at: str) -> list[dict]:
    """Rows of the ``+ HVG`` entry, credited and identified as its parent."""
    return task_rows(
        hvg_name(submission.manifest["model_name"]),
        result,
        submitted_at,
        submission.manifest["submission_id"],
        submission.credits,
    )


def _manifests(token: str | None) -> list[dict]:
    """Every archived submission manifest in the private results dataset."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi(token=token).list_repo_files(RESULTS_REPO, repo_type="dataset")
    paths = [
        path
        for path in files
        if path.startswith(ARCHIVE_PREFIX) and path.endswith(f"/{MANIFEST_FILENAME}")
    ]
    return [
        json.loads(
            Path(
                hf_hub_download(RESULTS_REPO, path, repo_type="dataset", token=token)
            ).read_text()
        )
        for path in paths
    ]


def _archived_file(manifest: dict, token: str | None) -> Path:
    """Download the embedding file a manifest describes."""
    from huggingface_hub import hf_hub_download

    filename = (
        f"{ARCHIVE_PREFIX}{manifest['submission_id']}/"
        f"submission{manifest['source_format'] or '.bin'}"
    )
    return Path(
        hf_hub_download(RESULTS_REPO, filename, repo_type="dataset", token=token)
    )


def _score(
    submission: LiveSubmission,
    reference: Path,
    checkpoint_dir: Path | None,
    token: str | None,
) -> dict:
    """Score one archived submission concatenated with the reference."""
    model = submission.manifest["model_name"]
    return score_all(
        _archived_file(submission.manifest, token),
        token,
        modality=submission.manifest["modality"],
        representation=GENE_SPACE if model in GENE_SPACE_MODELS else EMBEDDING,
        checkpoint_dir=checkpoint_dir,
        reference=reference,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Local HVG features; defaults to the published reference.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Only these model names."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Also write the rows to this CSV."
    )
    parser.add_argument(
        "--publish", action="store_true", help="Append the rows to the leaderboard."
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    token = os.environ.get("HF_TOKEN")
    if args.publish and not token:
        raise SystemExit("--publish needs write access: set HF_TOKEN.")
    reference = args.reference or fetch_hvg_reference(token)
    task_ids = {str(task[TASK_ID]) for task in fetch_tasks_registry(token)}
    submissions = [
        submission
        for submission in live_submissions(
            read_results(token), _manifests(token), task_ids
        )
        if args.models is None or submission.manifest["model_name"] in args.models
    ]
    submitted_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
    rows = []
    for submission in submissions:
        name = hvg_name(submission.manifest["model_name"])
        try:
            result = _score(submission, reference, args.checkpoint_dir, token)
        except (SubmissionError, EvaluatorError) as error:
            logger.warning("Skipping %s: %s", name, error)
            continue
        logger.info(
            "%s (%s): %d/%d tasks, %s",
            name,
            submission.manifest["modality"],
            result["n_scored"],
            result["n_total"],
            json.dumps(result["categories"], default=str),
        )
        rows += hvg_rows(submission, result, submitted_at)
    if args.out is not None:
        pd.DataFrame(rows).to_csv(args.out, index=False)
    if args.publish:
        append_results(rows, token)
        logger.info("Published %d '%s' task scores.", len(rows), HVG_TAG)


if __name__ == "__main__":
    main()
