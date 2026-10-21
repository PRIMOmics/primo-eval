"""Rescore the live entries of a board version into the current version.

A new board version opens with an empty board, as its harness has scored nothing
yet. This maintainer tool fills it from the archived embedding files, so nobody
has to submit again. An entry is live when it holds the latest row of a slot of
the old version, on a task that still exists.

- Each live submission is scored again, plain and ``+ HVG``, at the current pins.
  Its rows keep the model name, submission id, credits and the archive's
  ``submitted_at``, and carry the current version.
- The ``HVG-1200-genes`` baseline is scored again from the reference features,
  which are its submission.
- An entry keeps its task set: the tasks its old rows covered that still exist,
  plus the tasks the new version adds. ``--all-tasks`` scores every task the file
  covers instead.
- The datasets the current manifest no longer lists are dropped from the file,
  because ``score_all`` would refuse the whole file.
- A submission is skipped when the current version already holds rows of its
  entry from the same time or later. A second run after the deploy then only
  picks up the submissions made in between.
- An entry that fails, or that gets no score on a task it held, stops the run
  before anything is published. ``--models`` leaves an entry out on purpose.

``--from legacy`` reads the rows written before versions existed.

    HF_TOKEN=... python rescore.py --from 0.1 --dry-run
    HF_TOKEN=... python rescore.py --from 0.1 --checkpoint-dir ckpt --out rows.csv
    HF_TOKEN=... python rescore.py --from 0.1 --checkpoint-dir ckpt --publish
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from evaluator import (
    DATASET_ID,
    EMBEDDING,
    GENE_SPACE,
    RESULTS_REPO,
    TASK_ID,
    EvaluatorError,
    SubmissionError,
    _norm_id,
    fetch_manifest,
    fetch_tasks_registry,
    load_submission,
    manifest_ids,
    score_all,
    scoreable_tasks,
)
from leaderboard import latest_only
from results import (
    ARCHIVE_DIR,
    ARCHIVE_MANIFEST,
    HVG_REFERENCE_MODEL,
    HVG_TAG,
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    IS_BASELINE,
    MODEL_NAME,
    OWNER,
    PAPER_LINK,
    SUBMISSION_ID,
    append_results,
    archive_paths,
    fetch_hvg_reference,
    hvg_name,
    read_results,
    task_rows,
    version_rows,
    with_baseline_flag,
    with_owner,
)
from versions import CURRENT, VERSIONS

LEGACY = "legacy"
LEGACY_GENE_SPACE = ("COMPASS",)
CREDITS = (OWNER, INSTITUTION, IS_AUTHOR_SUBMISSION, PAPER_LINK)


@dataclass(frozen=True)
class Carry:
    """One live entry to score again: an archived file, or the HVG baseline.

    ``archive`` is the file's path in the results repo, or None for the baseline,
    whose file is the reference. ``task_ids`` None means every task of the file.
    """

    model_name: str
    is_baseline: bool
    submission_id: str
    submitted_at: str
    modality: str | None
    representation: str
    archive: str | None
    task_ids: frozenset[str] | None
    credits: dict = field(default_factory=dict, compare=False)


def representation_of(manifest: dict) -> str:
    """How to read an archive: from its manifest, or from the legacy list."""
    if "representation" in manifest:
        return manifest["representation"]
    return GENE_SPACE if manifest["model_name"] in LEGACY_GENE_SPACE else EMBEDDING


def _held_since(rows: pd.DataFrame, model: str, is_baseline: bool, since: str) -> bool:
    """Whether ``rows`` hold the entry ``model`` from ``since`` or later."""
    flagged = with_baseline_flag(rows)
    own = flagged[
        (flagged[MODEL_NAME] == model) & (flagged[IS_BASELINE] == is_baseline)
    ]
    return bool((own["submitted_at"].astype(str) >= since).any())


def _task_set(
    rows: pd.DataFrame, added: set[str], all_tasks: bool
) -> frozenset[str] | None:
    """The tasks ``rows`` cover plus the added ones, or None for every task."""
    if all_tasks:
        return None
    return frozenset(rows[TASK_ID].map(_norm_id)) | frozenset(added)


def carried(
    old: pd.DataFrame,
    new: pd.DataFrame,
    manifests: list[dict],
    task_ids: set[str],
    added: set[str],
    all_tasks: bool = False,
) -> tuple[list[Carry], list[str]]:
    """The live entries of ``old`` to score into ``new``, and those with no archive."""
    latest = with_owner(latest_only(old))
    if latest.empty:
        return [], []
    live = latest[latest[TASK_ID].map(_norm_id).isin(task_ids)]
    plain = live[
        ~live[IS_BASELINE] & ~live[MODEL_NAME].astype(str).str.endswith(HVG_TAG)
    ]
    carries = []
    for manifest in manifests:
        model, submission_id = manifest["model_name"], manifest["submission_id"]
        rows = plain[
            (plain[SUBMISSION_ID] == submission_id) & (plain[MODEL_NAME] == model)
        ]
        if rows.empty or _held_since(new, model, False, manifest["submitted_at"]):
            continue
        parent = rows.iloc[-1]
        carries.append(
            Carry(
                model_name=model,
                is_baseline=False,
                submission_id=submission_id,
                submitted_at=manifest["submitted_at"],
                modality=manifest["modality"],
                representation=representation_of(manifest),
                archive=archive_paths(submission_id, manifest["source_format"])[0],
                task_ids=_task_set(rows, added, all_tasks),
                credits={key: parent[key] for key in CREDITS},
            )
        )
    archived = {(m["model_name"], m["submission_id"]) for m in manifests}
    orphans = sorted(
        {
            row[MODEL_NAME]
            for _, row in plain.iterrows()
            if (row[MODEL_NAME], row[SUBMISSION_ID]) not in archived
            and not _held_since(new, row[MODEL_NAME], False, str(row["submitted_at"]))
        }
    )
    baseline = live[live[IS_BASELINE] & (live[MODEL_NAME] == HVG_REFERENCE_MODEL)]
    if not baseline.empty:
        since = str(baseline["submitted_at"].astype(str).max())
        if not _held_since(new, HVG_REFERENCE_MODEL, True, since):
            carries.append(
                Carry(
                    model_name=HVG_REFERENCE_MODEL,
                    is_baseline=True,
                    submission_id="",
                    submitted_at=since,
                    modality=None,
                    representation=GENE_SPACE,
                    archive=None,
                    task_ids=_task_set(baseline, added, all_tasks),
                )
            )
    return carries, orphans


def without_retired(path: Path, valid: set[str], out: Path) -> tuple[Path, list[str]]:
    """The file without the datasets ``valid`` no longer lists, and their ids.

    The file is rewritten to ``out`` only when a dataset has to go.
    """
    blocks = load_submission(path)
    retired = sorted(set(blocks) - valid)
    if not retired:
        return path, []
    kept = [
        block.assign(**{DATASET_ID: dataset_id})
        for dataset_id, block in blocks.items()
        if dataset_id in valid
    ]
    if not kept:
        raise SubmissionError(f"no dataset of the file is still listed: {retired}")
    frame = pd.concat(kept, ignore_index=True)
    frame.columns = frame.columns.map(str)
    frame.to_parquet(out, index=False)
    return out, retired


def rescore(
    carry: Carry,
    source: Path,
    reference: Path,
    datasets: list[dict],
    registry: list[dict],
    workdir: Path,
    token: str | None,
    checkpoint_dir: Path | None = None,
) -> list[dict]:
    """The rows of ``carry`` in the current version, plain and ``+ HVG``."""
    name = carry.submission_id or carry.model_name
    path, retired = without_retired(
        source, manifest_ids(datasets), workdir / f"{name}.parquet"
    )
    if retired:
        logging.warning(
            "%s: dropped the retired datasets %s", carry.model_name, retired
        )
    tasks = [
        task
        for task in registry
        if carry.task_ids is None or _norm_id(task[TASK_ID]) in carry.task_ids
    ]

    def _rows(model: str, features: Path | None) -> list[dict]:
        result = score_all(
            path,
            token,
            modality=carry.modality,
            representation=carry.representation,
            datasets=datasets,
            tasks=tasks,
            checkpoint_dir=checkpoint_dir,
            reference=features,
        )
        for invalid in result["invalid"]:
            logging.warning(
                "%s: no score on %s: %s", model, invalid[TASK_ID], invalid["reason"]
            )
        return task_rows(
            model, result, carry.submitted_at, carry.submission_id, carry.credits
        )

    if carry.is_baseline:
        return [{**row, IS_BASELINE: True} for row in _rows(carry.model_name, None)]
    return _rows(carry.model_name, None) + _rows(hvg_name(carry.model_name), reference)


def lost_tasks(carry: Carry, rows: list[dict], added: set[str]) -> list[str]:
    """The tasks ``carry`` held that one of its entries got no row for."""
    if carry.task_ids is None:
        return []
    names = [carry.model_name]
    if not carry.is_baseline:
        names.append(hvg_name(carry.model_name))
    held = carry.task_ids - added
    lost = set()
    for name in names:
        lost |= held - {row[TASK_ID] for row in rows if row[MODEL_NAME] == name}
    return sorted(lost)


def _score_carries(
    carries: list[Carry],
    added: set[str],
    reference: Path,
    datasets: list[dict],
    registry: list[dict],
    token: str | None,
    checkpoint_dir: Path | None,
) -> tuple[list[dict], list[str]]:
    """The rows of every carry, and one line per carry that failed."""
    rows, failures = [], []
    with TemporaryDirectory() as workdir:
        for carry in carries:
            source = (
                reference if carry.archive is None else _download(carry.archive, token)
            )
            try:
                scored = rescore(
                    carry,
                    source,
                    reference,
                    datasets,
                    registry,
                    Path(workdir),
                    token,
                    checkpoint_dir,
                )
            except (SubmissionError, EvaluatorError) as error:
                failures.append(f"{carry.model_name}: {error}")
                continue
            lost = lost_tasks(carry, scored, added)
            if lost:
                failures.append(f"{carry.model_name}: no score on held tasks {lost}")
            logging.info("%s: %d rows", carry.model_name, len(scored))
            rows += scored
    return rows, failures


def _manifests(token: str | None) -> list[dict]:
    """Every archived submission manifest in the private results dataset."""
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(RESULTS_REPO, repo_type="dataset")
    return [
        json.loads(_download(path, token).read_text())
        for path in files
        if path.startswith(f"{ARCHIVE_DIR}/") and path.endswith(f"/{ARCHIVE_MANIFEST}")
    ]


def _download(path_in_repo: str, token: str | None) -> Path:
    """One file of the private results dataset, at its latest commit."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(RESULTS_REPO, path_in_repo, repo_type="dataset", token=token)
    )


def _task_ids(registry: list[dict], datasets: list[dict]) -> set[str]:
    """The ids of the tasks of ``registry`` on the datasets the manifest lists."""
    return {
        _norm_id(task[TASK_ID])
        for task in scoreable_tasks(registry, manifest_ids(datasets))
    }


def _describe(carry: Carry) -> str:
    tasks = "every task" if carry.task_ids is None else f"{len(carry.task_ids)} tasks"
    kind = "baseline" if carry.is_baseline else carry.modality
    return f"{carry.model_name} ({kind}, {carry.representation}): {tasks}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from",
        dest="source",
        required=True,
        help=f"the version to rescore, or '{LEGACY}' for rows without one",
    )
    parser.add_argument("--models", nargs="+", default=None, help="only these models")
    parser.add_argument(
        "--all-tasks", action="store_true", help="score every task each file covers"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the rows here")
    parser.add_argument("--dry-run", action="store_true", help="list, score nothing")
    parser.add_argument(
        "--publish", action="store_true", help="append the rows to the leaderboard"
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = os.environ.get("HF_TOKEN")
    if args.publish and not token:
        raise SystemExit("--publish needs write access: set HF_TOKEN.")
    if args.source == CURRENT.number:
        raise SystemExit(f"{CURRENT.number} is the current version: pick an older one.")
    old_version = next((v for v in VERSIONS if v.number == args.source), None)
    if args.source != LEGACY and old_version is None:
        raise SystemExit(f"unknown version {args.source}")

    datasets = fetch_manifest(token)
    registry = fetch_tasks_registry(token)
    task_ids = _task_ids(registry, datasets)
    old_ids = (
        task_ids
        if old_version is None
        else _task_ids(
            fetch_tasks_registry(token, old_version.labels),
            fetch_manifest(token, old_version.data),
        )
    )
    added = task_ids - old_ids
    results = read_results(token)
    carries, orphans = carried(
        version_rows(results, "" if old_version is None else old_version.number),
        version_rows(results, CURRENT.number),
        _manifests(token),
        task_ids,
        added,
        args.all_tasks,
    )
    carries = [c for c in carries if args.models is None or c.model_name in args.models]
    logging.info("%d entries to rescore into %s:", len(carries), CURRENT.number)
    for carry in carries:
        logging.info("  %s", _describe(carry))
    if orphans:
        logging.warning("no archive, so not carried over: %s", orphans)
    if args.dry_run:
        return

    rows, failures = _score_carries(
        carries,
        added,
        fetch_hvg_reference(token),
        datasets,
        registry,
        token,
        args.checkpoint_dir,
    )
    if args.out is not None:
        pd.DataFrame(rows).to_csv(args.out, index=False)
    if failures:
        raise SystemExit(
            f"{len(failures)} entries failed, so nothing is published. Fix them, "
            "or leave them out with --models:\n" + "\n".join(failures)
        )
    if args.publish:
        append_results(rows, token)
        logging.info("published %d rows into version %s", len(rows), CURRENT.number)


if __name__ == "__main__":
    main()
