"""Score one task: the probe over frozen CV folds, or over a fixed transfer split."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_backend

from primomics.errors import EvaluatorError, SubmissionError
from primomics.metrics import METRICS, TaskScore
from primomics.probe import INNER_CV_JOBS, LOGREG_TOL, fit_predict
from primomics.submission import DATASET_ID, SAMPLE_ID, norm_id

SUBJECT_ID = "subject_id"
TASK_ID = "task_id"
LABEL = "label"
FOLD = "fold"
FOLD_PREFIX = f"{FOLD}_"
SPLIT = "split"
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"

OUTER_CV_JOBS = min(10, cpu_count())


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
