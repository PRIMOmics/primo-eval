"""The fixed linear probe fitted on a submission's embeddings.

Standardize, then a logistic regression (classification) or a ridge (regression).
The penalty is picked by an inner cross-validation grouped by patient, so the
samples of one patient never sit on both sides of an inner split.
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from primomics.errors import SubmissionError

RIDGE_ALPHAS = np.logspace(-6.0, 6.0, 25)
LOGREG_CS = np.logspace(-6.0, 6.0, 13)
INNER_CV = 5
INNER_CV_JOBS = 4
LOGREG_TOL = 1e-3
MAX_ITER = 5000
RANDOM_STATE = 0


def _group_inner_cv(groups_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject-grouped inner folds for ridge penalty selection."""
    n_groups = len(np.unique(groups_train))
    n_splits = min(INNER_CV, n_groups)
    if n_splits < 2:
        raise SubmissionError("an outer training fold has fewer than two subjects")
    splitter = GroupKFold(n_splits=n_splits)
    rows = np.zeros(len(groups_train))
    return list(splitter.split(rows, groups=groups_train))


def _ridge_fold_losses(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Return validation MSE for every alpha from one exact SVD."""
    return np.mean(_ridge_fold_target_losses(x_train, y_train, x_test, y_test), axis=1)


def _ridge_fold_target_losses(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Return per-target validation MSE while sharing one factorization."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_test = y_test[:, None]
    x_offset = np.mean(x_train, axis=0)
    y_offset = np.mean(y_train, axis=0)
    x_centered = x_train - x_offset
    y_centered = y_train - y_offset
    u, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    projected_x = (x_test - x_offset) @ vt.T
    projected_y = u.T @ y_centered
    shrinkage = singular_values[:, None] / (
        singular_values[:, None] ** 2 + RIDGE_ALPHAS[None, :]
    )
    losses = []
    for index in range(len(RIDGE_ALPHAS)):
        predictions = projected_x @ (shrinkage[:, index, None] * projected_y) + y_offset
        losses.append(np.mean((predictions - y_test) ** 2, axis=0))
    return np.asarray(losses)


def _ridge_alpha(matrix: np.ndarray, target: np.ndarray, groups: np.ndarray) -> float:
    """Select the grouped-CV ridge penalty without refitting for every alpha."""
    losses = [
        _ridge_fold_losses(matrix[train], target[train], matrix[test], target[test])
        for train, test in _group_inner_cv(groups)
    ]
    return float(RIDGE_ALPHAS[np.argmin(np.mean(losses, axis=0))])


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    groups_train: np.ndarray,
) -> np.ndarray:
    """Fit ridge with an exact grouped-CV spectral alpha search."""
    alpha = _ridge_alpha(x_train, y_train, groups_train)
    return Ridge(alpha=alpha).fit(x_train, y_train).predict(x_test)


def _relabelled_groups(groups: np.ndarray, seed: int) -> np.ndarray:
    """Reorder group tie-breaks without shuffling label counts between groups."""
    unique, codes = np.unique(groups, return_inverse=True)
    return np.random.default_rng(seed).permutation(len(unique))[codes]


def _inner_cv(
    y_train: np.ndarray, groups_train: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified subject-grouped folds for logistic penalty selection."""
    classes = np.unique(y_train)
    support = min(len(np.unique(groups_train[y_train == label])) for label in classes)
    n_splits = min(INNER_CV, support)
    if n_splits < 2:
        raise SubmissionError(
            "an outer training fold has fewer than two subjects in a class"
        )
    rows = np.zeros(len(y_train))
    relabelled_groups = _relabelled_groups(groups_train, RANDOM_STATE)
    splitter = StratifiedGroupKFold(n_splits=n_splits)
    splits = list(splitter.split(rows, y_train, relabelled_groups))
    expected_classes = set(classes)
    if not all(
        set(np.unique(y_train[train])) == expected_classes
        and set(np.unique(y_train[test])) == expected_classes
        for train, test in splits
    ):
        raise SubmissionError(
            "could not construct inner subject-grouped folds with every class"
        )
    return splits


def fit_predict(
    task_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classes: np.ndarray,
    groups_train: np.ndarray,
    inner_cv_jobs: int = INNER_CV_JOBS,
    logreg_tol: float | None = None,
) -> np.ndarray:
    """Fit the fixed linear probe on train, return predictions for test."""
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    logreg_tol = LOGREG_TOL if logreg_tol is None else logreg_tol
    if task_type == "classification":
        if len(np.unique(y_train)) < 2:
            raise SubmissionError("a training fold has a single class")
        scoring = "roc_auc" if len(classes) == 2 else "roc_auc_ovr_weighted"
        clf = LogisticRegressionCV(
            Cs=LOGREG_CS,
            cv=_inner_cv(y_train, groups_train),
            scoring=scoring,
            solver="lbfgs",
            max_iter=MAX_ITER,
            n_jobs=inner_cv_jobs,
            random_state=RANDOM_STATE,
            tol=logreg_tol,
        ).fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        col = {c: i for i, c in enumerate(clf.classes_)}
        out = np.zeros((x_test.shape[0], len(classes)))
        for j, c in enumerate(classes):
            if c in col:
                out[:, j] = proba[:, col[c]]
        return out
    return ridge_predict(x_train, y_train, x_test, groups_train)
