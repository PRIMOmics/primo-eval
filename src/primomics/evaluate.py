"""Score a submission: each task with the probe, then per-family means and coverage.

A task is scored over its frozen CV folds, or over a fixed transfer split. Binary
tasks with fewer than 30 subjects use leave-pair-out over subject pairs instead. A
perturbation task decodes a hidden response matrix instead of one label. The
caller passes the datasets, the tasks and functions that return a task's labels
and responses, so this module never knows where they are stored.
"""

import itertools
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_backend
from sklearn.preprocessing import StandardScaler

from primomics.errors import EvaluatorError, SubmissionError
from primomics.metrics import (
    METRICS,
    TaskScore,
    category_means,
    compute_residual_sample_spearman,
    compute_target_centered_sample_spearman,
)
from primomics.probe import INNER_CV_JOBS, LOGREG_TOL, fit_predict, ridge_predict
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
PAIR_ID = "pair_id"
PAIRING = "pairing"
PAIRED = "paired"
PERTURBATION = "perturbation"
RESIDUAL_SPEARMAN = "residual_spearman"
CENTERED_SPEARMAN = "centered_spearman"
MODALITIES = {"bulk-rna": "bulk RNA", "single-cell-rna": "Single Cell RNA"}

EMBEDDING = "embedding"
GENE_SPACE = "gene_space"
REPRESENTATIONS = (EMBEDDING, GENE_SPACE)

OUTER_CV_JOBS = min(10, cpu_count())
OUTER_PARALLEL_MIN_SAMPLES = 30
MIN_CV_SUBJECTS = 30
PERTURBATION_TEST_Z_CLIP = 20.0
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


@dataclass(frozen=True)
class PerturbationTargets:
    """Hidden response vectors and their aligned gene identifiers."""

    delta: np.ndarray
    gene_ids: np.ndarray


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
    is_perturbation = PAIR_ID in df.columns
    required = (
        {SAMPLE_ID, SUBJECT_ID, PAIR_ID}
        if is_perturbation
        else {SAMPLE_ID, SUBJECT_ID, LABEL}
    )
    missing = required - set(df.columns)
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


def load_targets(path: str | Path) -> PerturbationTargets:
    """Read and validate one task's hidden perturbation responses."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"delta", "gene_ids"} - set(data.files)
        if missing:
            raise EvaluatorError(f"targets.npz missing arrays: {sorted(missing)}")
        delta = np.asarray(data["delta"], dtype=float)
        gene_ids = np.asarray(data["gene_ids"]).astype(str)
    if delta.ndim != 2:
        raise EvaluatorError(f"targets delta must be 2D, got shape {delta.shape}")
    if gene_ids.ndim != 1 or len(gene_ids) != delta.shape[1]:
        raise EvaluatorError("targets gene_ids must align with delta columns")
    if not np.isfinite(delta).all():
        raise EvaluatorError("targets delta contains NaN or inf values")
    return PerturbationTargets(delta=delta, gene_ids=gene_ids)


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


def _load_reference(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read the gene-space reference concatenated to submissions, by dataset.

    The reference is ours, so any defect in it is an ``EvaluatorError``.
    """
    try:
        reference = {
            dataset_id: to_embedding_frame(block)
            for dataset_id, block in load_submission(path).items()
        }
    except (OSError, ValueError) as error:
        message = f"could not load the reference features: {error}"
        raise EvaluatorError(message) from error
    non_finite = sorted(
        dataset_id
        for dataset_id, frame in reference.items()
        if not np.isfinite(frame.to_numpy()).all()
    )
    if non_finite:
        raise EvaluatorError(f"reference features hold NaN or inf values: {non_finite}")
    return reference


def _concat_reference(emb: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Append the reference columns to each sample, then L2-normalize the row.

    Samples absent from either side are dropped, so a labelled sample the
    submission lacks still fails ``_align`` exactly as without a reference. One
    the reference lacks is caught first by ``_check_reference_covers``.
    """
    joined = emb.add_prefix("model_").join(
        reference.add_prefix("reference_"), how="inner"
    )
    return l2_normalize_rows(joined)


def _check_reference_covers(
    labels: pd.DataFrame, reference: pd.DataFrame, task_id: str
) -> None:
    """Raise if the reference lacks a labelled sample: that defect is ours."""
    lab_ids = [norm_id(v) for v in labels[SAMPLE_ID]]
    present = set(reference.index)
    missing = [s for s in lab_ids if s not in present]
    if missing:
        raise EvaluatorError(
            f"reference features lack {len(missing)}/{len(lab_ids)} labelled "
            f"samples of {task_id}, e.g. {missing[:5]}"
        )


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


def _uses_leave_pair_out(metric: str, labels: pd.DataFrame, groups: np.ndarray) -> bool:
    """Is this a binary AUROC task with fewer than ``MIN_CV_SUBJECTS`` subjects?"""
    return (
        metric == "auroc"
        and labels[LABEL].nunique() == 2
        and len(np.unique(groups)) < MIN_CV_SUBJECTS
    )


def _subject_labels(y: np.ndarray, subjects: np.ndarray) -> pd.Series:
    """Return one label per subject, indexed by subject id."""
    per_subject = pd.Series(y).groupby(subjects)
    if (per_subject.nunique() > 1).any():
        raise EvaluatorError("a subject carries more than one label")
    return per_subject.first()


def _ranked_pairs(scores: np.ndarray, is_positive: np.ndarray) -> tuple[float, int]:
    """Count correctly ordered positive-negative sample pairs, ties as half."""
    gaps = scores[is_positive][:, None] - scores[~is_positive][None, :]
    if not np.isfinite(gaps).all():
        raise SubmissionError("a leave-pair-out fit produced a non-finite score")
    return float((gaps > 0).sum() + 0.5 * (gaps == 0).sum()), int(gaps.size)


def _leave_pair_out_score(
    y: np.ndarray, groups: np.ndarray, matrix: np.ndarray, classes: np.ndarray
) -> float:
    """Fraction of sample pairs ranked right, each subject pair held out once.

    The probe is fit on every other subject, with its inner penalty search
    grouped by subject as in CV.
    """
    per_subject = _subject_labels(y, groups)
    correct, total = 0.0, 0
    for positive, negative in itertools.product(
        per_subject.index[per_subject == classes[1]],
        per_subject.index[per_subject == classes[0]],
    ):
        held_out = np.isin(groups, [positive, negative])
        scores = fit_predict(
            "classification",
            matrix[~held_out],
            y[~held_out],
            matrix[held_out],
            classes,
            groups[~held_out],
        )[:, 1]
        pair_correct, pair_total = _ranked_pairs(scores, groups[held_out] == positive)
        correct += pair_correct
        total += pair_total
    return correct / total


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


def _select_degs(delta_train: np.ndarray, n_top: int) -> np.ndarray:
    """Select the strongest mean absolute responses using training rows only."""
    n_keep = min(int(n_top), delta_train.shape[1])
    if n_keep <= 0:
        raise EvaluatorError("perturbation task has no response genes to select")
    effect = np.abs(np.mean(delta_train, axis=0))
    return np.argsort(effect, kind="stable")[-n_keep:]


def _mean_scores(scores: list[float]) -> float:
    """Average finite fold scores."""
    finite = np.asarray(scores)[np.isfinite(scores)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _perturbation_fold(
    matrix: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    n_top: int,
    target_centered: bool,
) -> float:
    """Fit and score one leakage-safe perturbation fold."""
    if not train.any() or not test.any():
        raise EvaluatorError("perturbation fold has an empty train or test side")
    genes = _select_degs(delta[train], n_top)
    y_train = delta[train][:, genes]
    y_test = delta[test][:, genes]
    scaler = StandardScaler().fit(matrix[train])
    x_train = scaler.transform(matrix[train])
    x_test = np.clip(
        scaler.transform(matrix[test]),
        -PERTURBATION_TEST_Z_CLIP,
        PERTURBATION_TEST_Z_CLIP,
    )
    predictions = ridge_predict(x_train, y_train, x_test, groups[train])
    if target_centered:
        correlation = compute_target_centered_sample_spearman(y_test, predictions)
    else:
        training_mean = np.mean(y_train, axis=0)
        correlation = compute_residual_sample_spearman(
            y_test, predictions, training_mean
        )
    if not np.isfinite(correlation):
        raise SubmissionError("a perturbation fold has a degenerate Spearman score")
    return float(np.clip(correlation, -1.0, 1.0))


def _perturbation_repeat(
    matrix: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    n_top: int,
) -> float:
    """Score every fold in one perturbation repeat."""
    return _mean_scores(
        [
            _perturbation_fold(matrix, delta, groups, train, test, n_top, False)
            for train, test in partitions
        ]
    )


def _parallel_perturbation_repeats(
    matrix: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    repeats: list[list[tuple[np.ndarray, np.ndarray]]],
    n_top: int,
) -> tuple[float, ...]:
    """Score independent perturbation repeats across available workers."""
    workers = (
        min(OUTER_CV_JOBS, len(repeats))
        if len(matrix) >= OUTER_PARALLEL_MIN_SAMPLES
        else 1
    )
    if workers == 1:
        scores = [
            _perturbation_repeat(matrix, delta, groups, folds, n_top)
            for folds in repeats
        ]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            scores = Parallel(n_jobs=workers)(
                delayed(_perturbation_repeat)(matrix, delta, groups, folds, n_top)
                for folds in repeats
            )
    return tuple(float(score) for score in scores)


def _paired_perturbation_scores(
    matrix: np.ndarray,
    delta: np.ndarray,
    labels: pd.DataFrame,
    n_top: int,
) -> tuple[float, ...]:
    """Score each frozen subject-grouped CV partition the labels carry."""
    groups = _subject_groups(labels)
    repeats = []
    for column in _fold_columns(labels):
        folds = labels[column].to_numpy()
        repeats.append(
            [(folds != fold, folds == fold) for fold in sorted(np.unique(folds))]
        )
    return _parallel_perturbation_repeats(matrix, delta, groups, repeats, n_top)


def _transfer_perturbation_scores(
    matrix: np.ndarray,
    delta: np.ndarray,
    labels: pd.DataFrame,
    n_top: int,
    target_centered: bool,
) -> tuple[tuple[float, ...], int]:
    """Train every response-decoding component on the source disease only."""
    groups = _subject_groups(labels)
    split = labels[SPLIT].astype(str).to_numpy()
    train, test = split == SPLIT_TRAIN, split == SPLIT_TEST
    metrics = _perturbation_fold(
        matrix, delta, groups, train, test, n_top, target_centered
    )
    return (metrics,), int(test.sum())


def _score_perturbation_task(
    task: dict,
    labels: pd.DataFrame,
    emb: pd.DataFrame,
    targets: PerturbationTargets,
) -> TaskScore:
    """Decode one hidden response matrix from submitted baseline embeddings."""
    if len(labels) != len(targets.delta):
        raise EvaluatorError("labels.csv and targets.npz have different row counts")
    matrix = _align(labels, emb)
    n_top = int(task.get("n_top_de_genes", 0))
    pairing = str(task.get(PAIRING, labels.get(PAIRING, pd.Series([PAIRED])).iloc[0]))
    if SPLIT in labels:
        repeat_scores, n_samples = _transfer_perturbation_scores(
            matrix,
            targets.delta,
            labels,
            n_top,
            bool(task.get("target_centered", False)),
        )
    elif pairing == PAIRED:
        repeat_scores = _paired_perturbation_scores(
            matrix, targets.delta, labels, n_top
        )
        n_samples = len(labels)
    else:
        raise EvaluatorError(f"unsupported perturbation pairing: {pairing}")
    return TaskScore(
        task_id=norm_id(task[TASK_ID]),
        dataset_id=norm_id(task[DATASET_ID]),
        category=str(task["category"]),
        metric=str(task.get("metric", RESIDUAL_SPEARMAN)),
        score=float(np.mean(repeat_scores)),
        n_samples=n_samples,
        repeat_scores=repeat_scores,
    )


def score_task(
    task: dict,
    labels: pd.DataFrame,
    emb: pd.DataFrame,
    targets: PerturbationTargets | None = None,
) -> TaskScore:
    """Score one task: repeated fold-wise CV, leave-pair-out, or a transfer split."""
    metric = task["metric"]
    if task.get("task_type") == PERTURBATION:
        if metric not in {CENTERED_SPEARMAN, RESIDUAL_SPEARMAN}:
            raise EvaluatorError(
                f"perturbation task {task.get(TASK_ID)} needs metric "
                f"'{CENTERED_SPEARMAN}'"
            )
        if targets is None:
            raise EvaluatorError("perturbation task is missing targets.npz")
        return _score_perturbation_task(task, labels, emb, targets)
    if metric not in METRICS:
        raise EvaluatorError(f"unknown metric '{metric}' for task {task.get(TASK_ID)}")
    task_type = task["task_type"]
    matrix = _align(labels, emb)
    y = labels[LABEL].to_numpy()
    y_eval = y
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
    elif _uses_leave_pair_out(metric, labels, groups):
        score = _leave_pair_out_score(y, groups, matrix, classes)
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
    reference: pd.DataFrame | None,
    fetch_labels: Callable[[str], pd.DataFrame],
    fetch_targets: Callable[[str], PerturbationTargets] | None,
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
    if reference is not None:
        _check_reference_covers(labels, reference, task_id)
    targets = None
    if task.get("task_type") == PERTURBATION:
        if fetch_targets is None:
            raise EvaluatorError("a perturbation task needs fetch_targets")
        try:
            targets = _retry(lambda: fetch_targets(task_id))
        except Exception as error:
            raise EvaluatorError(
                f"could not load perturbation targets for a task: {error}"
            ) from error
    try:
        score = score_task(task, labels, emb, targets)
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
    fetch_targets: Callable[[str], PerturbationTargets] | None = None,
    modality: str | None = None,
    representation: str = EMBEDDING,
    reference: str | Path | None = None,
) -> dict:
    """Score a submission per task and roll it up into a blind result.

    Unknown dataset ids and bad files raise/record a ``SubmissionError``; a
    failed fetch or internal error raises ``EvaluatorError`` (our side, not the
    submitter's). A dataset is embedded once and reused across all its tasks.

    ``EMBEDDING`` columns are put on the unit sphere first, ``GENE_SPACE`` ones
    keep their scale. A ``reference`` file (same layout as a submission, gene
    space) is then appended to every sample and the concatenated row put back
    on the unit sphere.
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

    reference_by_ds = _load_reference(reference) if reference is not None else {}

    emb_by_ds, block_error = {}, {}
    for dataset_id, block in blocks.items():
        try:
            frame = to_embedding_frame(block)
            if representation == EMBEDDING:
                frame = l2_normalize_rows(frame)
            if reference is not None:
                if dataset_id not in reference_by_ds:
                    raise EvaluatorError(
                        f"reference features lack dataset {dataset_id}"
                    )
                frame = _concat_reference(frame, reference_by_ds[dataset_id])
            emb_by_ds[dataset_id] = frame
        except SubmissionError as error:
            block_error[dataset_id] = str(error)

    outcomes = [
        _task_outcome(
            task,
            emb_by_ds.get(norm_id(task[DATASET_ID])),
            block_error.get(norm_id(task[DATASET_ID])),
            reference_by_ds.get(norm_id(task[DATASET_ID])),
            fetch_labels,
            fetch_targets,
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
