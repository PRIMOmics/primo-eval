"""Metrics that turn a probe's predictions into a task score.

AUROC for classification, Pearson for clinical scores, and a sample-wise
Spearman for perturbation responses.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray | None = None
) -> float:
    """AUROC from class probabilities (``y_pred`` is an ``(n, n_classes)`` matrix).

    ``classes`` names the columns of ``y_pred``. It matters whenever ``y_true``
    holds fewer classes than the task does -- a transfer split whose test cohort
    misses one. The columns of the absent classes are dropped and the rest
    renormalized, so the score never reads a column belonging to another class,
    and never trips sklearn's sum-to-one check.
    """
    classes = np.unique(y_true) if classes is None else np.asarray(classes)
    present = np.isin(classes, np.unique(y_true))
    scores = np.asarray(y_pred)[:, present]
    totals = scores.sum(axis=1, keepdims=True)
    if not (totals > 0).all():
        raise ValueError(
            "some samples carry no probability on any class present in y_true"
        )
    scores = scores / totals
    kept = classes[present]
    if len(kept) == 2:
        return float(roc_auc_score(y_true, scores[:, 1]))
    return float(
        roc_auc_score(
            y_true, scores, multi_class="ovr", average="weighted", labels=kept
        )
    )


def compute_pearson(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray | None = None
) -> float:
    """Pearson r between predictions and targets; NaN if either is constant."""
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return float("nan")
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return stable, one-based average ranks with deterministic tie handling."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def _mean_sample_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean row-wise rank correlation, scoring degenerate rows at zero."""
    correlations = []
    for truth, prediction in zip(y_true, y_pred):
        correlation = compute_pearson(_average_ranks(truth), _average_ranks(prediction))
        correlations.append(correlation if np.isfinite(correlation) else 0.0)
    return float(np.mean(correlations)) if correlations else float("nan")


def compute_residual_sample_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_mean: np.ndarray,
) -> float:
    """Mean sample-wise Spearman beyond the training-fold mean response."""
    return _mean_sample_spearman(y_true - training_mean, y_pred - training_mean)


def compute_target_centered_sample_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean sample-wise Spearman after removing each target-cohort gene mean."""
    truth_centered = y_true - np.mean(y_true, axis=0)
    prediction_centered = y_pred - np.mean(y_pred, axis=0)
    return _mean_sample_spearman(truth_centered, prediction_centered)


METRICS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]] = {
    "auroc": compute_auroc,
    "pearson": compute_pearson,
}


@dataclass(frozen=True)
class TaskScore:
    """One task's result: a raw metric plus the category it is grouped under."""

    task_id: str
    dataset_id: str
    category: str
    metric: str
    score: float
    n_samples: int
    repeat_scores: tuple[float, ...] = ()
    diagnostics: dict[str, float] = field(default_factory=dict)
