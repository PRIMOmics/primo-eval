"""Score a submission: each task with the probe, then per-family means and coverage.

A task is scored over its frozen CV folds, or over a fixed transfer split. The
caller passes the datasets, the tasks and a function that returns a task's
labels, so this module never knows where the labels are stored.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_backend

from primomics.errors import EvaluatorError, SubmissionError
from primomics.metrics import METRICS, TaskScore, category_means
from primomics.probe import INNER_CV_JOBS, LOGREG_TOL, fit_predict
from primomics.submission import (
    DATASET_ID,
    SAMPLE_ID,
    l2_normalize_rows,
    load_submission,
    norm_id,
    to_embedding_frame,
)

SUBJECT_ID = "subject_id"
TASK_ID = "task_id"
LABEL = "label"
FOLD = "fold"
FOLD_PREFIX = f"{FOLD}_"
SPLIT = "split"
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
MODALITIES = {"bulk-rna": "bulk RNA", "single-cell-rna": "Single Cell RNA"}

EMBEDDING = "embedding"
GENE_SPACE = "gene_space"
REPRESENTATIONS = (EMBEDDING, GENE_SPACE)

OUTER_CV_JOBS = min(10, cpu_count())
FETCH_ATTEMPTS = 3
FETCH_BACKOFF = 1.0

logger = logging.getLogger(__name__)


@dataclass
class TaskOutcome:
    """What happened to one scoreable task: scored / missing / invalid."""

    task_id: str
    dataset_id: str
    status: str
    score: TaskScore | None = None
    reason: str | None = None


def _fold_columns(labels: pd.DataFrame) -> list[str]:
    """Return the ordered external-CV fold columns in the labels."""
    repeated = [column for column in labels if column.startswith(FOLD_PREFIX)]
    invalid = [
        column for column in repeated if not column.removeprefix(FOLD_PREFIX).isdigit()
    ]
    if invalid:
        raise EvaluatorError(f"labels.csv has invalid repeat-fold columns: {invalid}")
    first = [FOLD] if FOLD in labels.columns else []
    return [
        *first,
        *sorted(repeated, key=lambda column: int(column.removeprefix(FOLD_PREFIX))),
    ]


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read a task's labels with repeated CV folds or one transfer split."""
    df = pd.read_csv(path)
    missing = {SAMPLE_ID, SUBJECT_ID, LABEL} - set(df.columns)
    if missing:
        raise EvaluatorError(f"labels.csv missing columns: {sorted(missing)}")
    _subject_groups(df)
    has_folds = bool(_fold_columns(df))
    has_split = SPLIT in df.columns
    if sum((has_folds, has_split)) != 1:
        raise EvaluatorError(
            "labels.csv needs exactly one evaluation assignment: folds or a "
            "transfer split."
        )
    return df


def _subject_groups(labels: pd.DataFrame) -> np.ndarray:
    """Return normalized subject ids, rejecting unusable metadata."""
    if SUBJECT_ID not in labels:
        raise EvaluatorError(f"labels.csv missing columns: ['{SUBJECT_ID}']")
    if labels[SUBJECT_ID].isna().any():
        raise EvaluatorError("labels.csv has missing subject_id values")
    groups = np.array([norm_id(value) for value in labels[SUBJECT_ID]])
    if np.any(groups == ""):
        raise EvaluatorError("labels.csv has empty subject_id values")
    return groups


def _align(labels: pd.DataFrame, emb: pd.DataFrame) -> np.ndarray:
    """Return the embedding matrix in labels order, or raise if a sample is missing."""
    lab_ids = [norm_id(v) for v in labels[SAMPLE_ID]]
    present = set(emb.index)
    missing = [s for s in lab_ids if s not in present]
    if missing:
        raise SubmissionError(
            f"{len(missing)}/{len(lab_ids)} labelled samples missing from "
            f"submission, e.g. {missing[:5]}"
        )
    matrix = emb.loc[lab_ids].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise SubmissionError("submission contains NaN or inf values")
    return matrix


def _fold_score(
    metric: str,
    task_type: str,
    y: np.ndarray,
    folds: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
    inner_cv_jobs: int = INNER_CV_JOBS,
    logreg_tol: float | None = None,
) -> float:
    """Return the mean native metric across one external CV partition."""
    scores = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        if len(np.unique(y[test])) < 2:
            raise EvaluatorError(
                f"CV fold {fold} holds a single label value; the folds are ours to fix"
            )
        predictions = fit_predict(
            task_type,
            matrix[~test],
            y[~test],
            matrix[test],
            classes,
            groups[~test],
            inner_cv_jobs,
            logreg_tol,
        )
        try:
            scores.append(METRICS[metric](y[test], predictions, classes))
        except ValueError as error:
            raise SubmissionError(
                f"CV fold {fold} is not scoreable: {error}"
            ) from error
    if not np.isfinite(scores).all():
        raise SubmissionError("a CV fold produced a non-finite score")
    return float(np.mean(scores))


def _parallel_fold_score(
    metric: str,
    task_type: str,
    y: np.ndarray,
    folds: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
    logreg_tol: float,
) -> float:
    """Score one repeat in an outer worker without nested inner processes."""
    return _fold_score(
        metric,
        task_type,
        y,
        folds,
        matrix,
        classes,
        groups,
        inner_cv_jobs=1,
        logreg_tol=logreg_tol,
    )


def _repeated_cv_scores(
    metric: str,
    task_type: str,
    y: np.ndarray,
    labels: pd.DataFrame,
    matrix: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, ...]:
    """Return one mean fold metric for every external CV repeat."""
    groups = _subject_groups(labels)
    columns = _fold_columns(labels)
    workers = min(OUTER_CV_JOBS, len(columns))
    if workers == 1:
        scores = [
            _fold_score(
                metric,
                task_type,
                y,
                labels[column].to_numpy(),
                matrix,
                classes,
                groups,
            )
            for column in columns
        ]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            scores = Parallel(n_jobs=workers)(
                delayed(_parallel_fold_score)(
                    metric,
                    task_type,
                    y,
                    labels[column].to_numpy(),
                    matrix,
                    classes,
                    groups,
                    LOGREG_TOL,
                )
                for column in columns
            )
    return tuple(float(score) for score in scores)


def _transfer_predict(
    task_type: str,
    y: np.ndarray,
    split: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on the train cohort, predict the test cohort (no pooling)."""
    train, test = split == SPLIT_TRAIN, split == SPLIT_TEST
    if not train.any() or not test.any():
        raise EvaluatorError("transfer labels.csv has an empty train or test side")
    if len(np.unique(y[test])) < 2:
        raise EvaluatorError(
            "the transfer test cohort holds a single label value; the split is ours "
            "to fix"
        )
    preds = fit_predict(
        task_type,
        matrix[train],
        y[train],
        matrix[test],
        classes,
        groups[train],
    )
    return y[test], preds


def score_task(task: dict, labels: pd.DataFrame, emb: pd.DataFrame) -> TaskScore:
    """Score one task: repeated fold-wise CV, or a fixed transfer split."""
    metric = task["metric"]
    if metric not in METRICS:
        raise EvaluatorError(f"unknown metric '{metric}' for task {task.get(TASK_ID)}")
    task_type = task["task_type"]
    matrix = _align(labels, emb)
    y = labels[LABEL].to_numpy()
    classes = np.unique(y) if task_type == "classification" else np.array([])
    groups = _subject_groups(labels)

    if SPLIT in labels.columns:
        y_eval, preds = _transfer_predict(
            task_type,
            y,
            labels[SPLIT].to_numpy(),
            matrix,
            classes,
            groups,
        )
        try:
            score = METRICS[metric](y_eval, preds, classes)
        except ValueError as error:
            raise SubmissionError(f"transfer task not scoreable: {error}") from error
        repeat_scores = (score,)
    else:
        repeat_scores = _repeated_cv_scores(
            task_type=task_type,
            metric=metric,
            y=y,
            labels=labels,
            matrix=matrix,
            classes=classes,
        )
        score = float(np.mean(repeat_scores))
        y_eval = y

    return TaskScore(
        task_id=norm_id(task[TASK_ID]),
        dataset_id=norm_id(task[DATASET_ID]),
        category=str(task["category"]),
        metric=metric,
        score=score,
        n_samples=len(y_eval),
        repeat_scores=repeat_scores,
    )


def manifest_ids(manifest: list[dict], modality: str | None = None) -> set[str]:
    """Return manifest dataset ids, optionally for one modality."""
    return {
        norm_id(entry["id"])
        for entry in manifest
        if modality is None or entry.get("modality") == modality
    }


def scoreable_tasks(tasks: list[dict], valid_ids: set[str]) -> list[dict]:
    """Tasks whose dataset is in the public manifest; orphans are skipped.

    Shared by ``score_all`` and the leaderboard so both agree on what full
    coverage means. An orphan (a task on a dataset not yet published) is logged
    and dropped rather than making full coverage unreachable for everyone.
    """
    kept, orphans = [], []
    for task in tasks:
        if norm_id(task[DATASET_ID]) in valid_ids:
            kept.append(task)
        else:
            orphans.append(norm_id(task[TASK_ID]))
    if orphans:
        logger.warning("skipping tasks on unpublished datasets: %s", orphans)
    return kept


def _retry(fn):
    """Call ``fn``, retrying transient failures with linear backoff."""
    for attempt in range(FETCH_ATTEMPTS):
        try:
            return fn()
        except Exception:
            if attempt == FETCH_ATTEMPTS - 1:
                raise
            time.sleep(FETCH_BACKOFF * (attempt + 1))


def _task_outcome(
    task: dict,
    emb: pd.DataFrame | None,
    block_error: str | None,
    fetch_labels: Callable[[str], pd.DataFrame],
) -> TaskOutcome:
    """Score one task, mapping each failure to the right category."""
    task_id = norm_id(task[TASK_ID])
    dataset_id = norm_id(task[DATASET_ID])
    if emb is None and block_error is None:
        return TaskOutcome(task_id, dataset_id, "missing")
    if block_error is not None:
        return TaskOutcome(task_id, dataset_id, "invalid", reason=block_error)
    try:
        labels = _retry(lambda: fetch_labels(task_id))
    except Exception as error:
        raise EvaluatorError(
            f"could not load evaluation data for a task: {error}"
        ) from error
    try:
        score = score_task(task, labels, emb)
    except SubmissionError as error:
        return TaskOutcome(task_id, dataset_id, "invalid", reason=str(error))
    if not np.isfinite(score.score):
        return TaskOutcome(
            task_id,
            dataset_id,
            "invalid",
            reason="degenerate score (constant predictions)",
        )
    return TaskOutcome(task_id, dataset_id, "scored", score=score)


def score_all(
    path: str | Path,
    *,
    datasets: list[dict],
    tasks: list[dict],
    fetch_labels: Callable[[str], pd.DataFrame],
    modality: str | None = None,
    representation: str = EMBEDDING,
) -> dict:
    """Score a submission per task and roll it up into a blind result.

    Unknown dataset ids and bad files raise/record a ``SubmissionError``; a
    failed fetch or internal error raises ``EvaluatorError`` (our side, not the
    submitter's). A dataset is embedded once and reused across all its tasks.

    ``EMBEDDING`` columns are put on the unit sphere first, ``GENE_SPACE`` ones
    keep their scale.
    """
    if representation not in REPRESENTATIONS:
        raise SubmissionError(
            f"unknown representation '{representation}'; expected one of "
            f"{list(REPRESENTATIONS)}"
        )
    blocks = load_submission(path)
    if modality is not None and modality not in MODALITIES:
        raise SubmissionError(f"unknown or unavailable modality: {modality}")
    all_ids = manifest_ids(datasets)
    valid = manifest_ids(datasets, MODALITIES.get(modality))
    if modality is not None and not valid:
        raise SubmissionError(f"unknown or unavailable modality: {modality}")

    unknown = sorted(set(blocks) - all_ids)
    if unknown:
        raise SubmissionError(f"unknown dataset_id(s) not in the benchmark: {unknown}")
    wrong_modality = sorted(set(blocks) - valid)
    if wrong_modality:
        raise SubmissionError(
            f"dataset_id(s) outside the selected modality: {wrong_modality}"
        )

    emb_by_ds, block_error = {}, {}
    for dataset_id, block in blocks.items():
        try:
            frame = to_embedding_frame(block)
            if representation == EMBEDDING:
                frame = l2_normalize_rows(frame)
            emb_by_ds[dataset_id] = frame
        except SubmissionError as error:
            block_error[dataset_id] = str(error)

    outcomes = [
        _task_outcome(
            task,
            emb_by_ds.get(norm_id(task[DATASET_ID])),
            block_error.get(norm_id(task[DATASET_ID])),
            fetch_labels,
        )
        for task in scoreable_tasks(tasks, all_ids)
        if norm_id(task[DATASET_ID]) in valid
    ]
    return _summarize(outcomes)


def _summarize(outcomes: list[TaskOutcome]) -> dict:
    """Combine per-task outcomes into blind per-category numbers + coverage."""
    scored = [o.score for o in outcomes if o.status == "scored"]
    dataset_status = _dataset_status(outcomes)
    n_scored, n_total = len(scored), len(outcomes)
    n_ds_scored = sum(1 for status in dataset_status.values() if status == "scored")
    return {
        "per_task": scored,
        "categories": category_means(scored),
        "invalid": [
            {"task_id": o.task_id, "dataset_id": o.dataset_id, "reason": o.reason}
            for o in outcomes
            if o.status == "invalid"
        ],
        "missing": sorted({o.dataset_id for o in outcomes if o.status == "missing"}),
        "incomplete": sorted(
            ds for ds, status in dataset_status.items() if status == "incomplete"
        ),
        "coverage": n_scored / n_total if n_total else 0.0,
        "n_scored": n_scored,
        "n_total": n_total,
        "full_coverage": n_total > 0 and n_scored == n_total,
        "n_datasets_scored": n_ds_scored,
        "n_datasets_total": len(dataset_status),
    }


def _dataset_status(outcomes: list[TaskOutcome]) -> dict[str, str]:
    """Blind per-dataset STATUS only (no score): scored / missing / incomplete."""
    statuses: dict[str, set[str]] = defaultdict(set)
    for outcome in outcomes:
        statuses[outcome.dataset_id].add(outcome.status)
    out = {}
    for dataset_id in sorted(statuses):
        seen = statuses[dataset_id]
        if seen == {"missing"}:
            out[dataset_id] = "missing"
        elif seen == {"scored"}:
            out[dataset_id] = "scored"
        else:
            out[dataset_id] = "incomplete"
    return out
