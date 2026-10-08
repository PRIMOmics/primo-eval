import numpy as np
import pytest
from sklearn.linear_model import RidgeCV

from primomics import probe
from primomics.errors import SubmissionError
from primomics.metrics import compute_auroc, compute_pearson
from primomics.probe import (
    INNER_CV,
    RIDGE_ALPHAS,
    _group_inner_cv,
    _inner_cv,
    _ridge_alpha,
    fit_predict,
)


def test_regression_inner_selection_keeps_subjects_on_one_side():
    groups = np.repeat([f"p{i}" for i in range(6)], 2)

    for train, test in _group_inner_cv(groups):
        assert set(groups[train]).isdisjoint(groups[test])


def test_spectral_ridge_matches_sklearn_grouped_cv():
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(48, 12))
    target = matrix @ rng.normal(size=12) + rng.normal(scale=0.2, size=48)
    groups = np.repeat([f"p{i}" for i in range(24)], 2)
    cv = _group_inner_cv(groups)
    expected = RidgeCV(
        alphas=RIDGE_ALPHAS,
        cv=cv,
        scoring="neg_mean_squared_error",
    ).fit(matrix, target)

    assert _ridge_alpha(matrix, target, groups) == expected.alpha_


def test_classification_inner_selection_is_subject_grouped():
    """The penalty selection never splits a patient across inner folds."""
    groups = np.repeat([f"p{i}" for i in range(8)], 2)
    y = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 2)
    cv = _inner_cv(y, groups)

    assert len(cv) == 4
    for train, test in cv:
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(y[test]) == {0, 1}


def test_classification_inner_selection_stratifies_real_labels():
    rng = np.random.default_rng(0)
    samples_per_subject = rng.integers(1, 5, size=105)
    groups = np.repeat(
        [f"donor-{index:03d}" for index in range(len(samples_per_subject))],
        samples_per_subject,
    )
    y = np.repeat(np.arange(len(samples_per_subject)) % 3 == 0, samples_per_subject)
    y = y.astype(int)
    ideal_positives = y.sum() / INNER_CV
    ideal_size = len(y) / INNER_CV

    for _, test in _inner_cv(y, groups):
        assert abs(y[test].sum() - ideal_positives) <= 2
        assert abs(len(test) - ideal_size) <= 3


def test_inner_folds_match_the_live_evaluator():
    """The exact inner splits the live evaluator makes for 12 patients."""
    subjects = np.arange(12)
    sizes = 1 + subjects % 3
    groups = np.repeat([f"p{s:02d}" for s in subjects], sizes)
    y = np.repeat(subjects % 2, sizes)

    logistic = [sorted(test.tolist()) for _, test in _inner_cv(y, groups)]
    ridge = [sorted(test.tolist()) for _, test in _group_inner_cv(groups)]

    assert logistic == [
        [13, 14, 15, 16, 17],
        [3, 4, 5, 6, 18],
        [9, 10, 11, 19, 20],
        [0, 12, 21, 22, 23],
        [1, 2, 7, 8],
    ]
    assert ridge == [
        [7, 8, 21, 22, 23],
        [1, 2, 15, 16, 17],
        [6, 9, 10, 11, 18],
        [0, 3, 4, 5, 12],
        [13, 14, 19, 20],
    ]


def test_probe_settings_match_the_live_evaluator(monkeypatch):
    """The protocol is these numbers: changing one changes every score."""
    seen = {}
    real = probe.LogisticRegressionCV

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(probe, "LogisticRegressionCV", spy)
    x, y, groups, train = _fixed_dataset("binary")
    fit_predict(
        "classification", x[train], y[train], x[~train], np.unique(y), groups[train]
    )

    assert np.array_equal(seen["Cs"], np.logspace(-6.0, 6.0, 13))
    assert np.array_equal(RIDGE_ALPHAS, np.logspace(-6.0, 6.0, 25))
    assert INNER_CV == 5
    assert (
        seen["scoring"],
        seen["solver"],
        seen["tol"],
        seen["max_iter"],
        seen["random_state"],
    ) == ("roc_auc", "lbfgs", 1e-3, 5000, 0)


def _fixed_dataset(kind):
    """A small dataset built from formulas, so it is the same on every machine."""
    rows = np.arange(60)
    subject = rows // 2
    groups = np.array([f"p{s:02d}" for s in subject])
    latent = np.sin(0.9 * subject)
    x = np.outer(latent, np.linspace(1.0, 0.2, 8)) + 0.8 * np.sin(
        np.outer(rows + 1, np.arange(1, 9))
    )
    if kind == "binary":
        y = (latent > 0).astype(int)
    elif kind == "multiclass":
        y = np.digitize(latent, [-0.5, 0.5])
    else:
        y = latent + 0.3 * np.cos(2.0 * rows)
    return x, y, groups, subject < 20


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("binary", 0.9791666666666666),
        ("multiclass", 0.8922619047619047),
        ("regression", 0.9613594026339882),
    ],
)
def test_matches_the_live_evaluator_on_a_fixed_dataset(kind, expected):
    """Scores the live evaluator gives on this data: a changed probe moves them."""
    x, y, groups, train = _fixed_dataset(kind)
    if kind == "regression":
        pred = fit_predict(
            kind, x[train], y[train], x[~train], np.array([]), groups[train]
        )
        score = compute_pearson(y[~train], pred)
    else:
        classes = np.unique(y)
        pred = fit_predict(
            "classification", x[train], y[train], x[~train], classes, groups[train]
        )
        score = compute_auroc(y[~train], pred, classes)

    assert score == pytest.approx(expected, abs=1e-9)


def _signal_or_noise(task_type, signal):
    rng = np.random.default_rng(3)
    groups = np.repeat([f"p{i:03d}" for i in range(100)], 2)
    target = np.repeat(rng.normal(size=100), 2)
    y = (target > 0).astype(int) if task_type == "classification" else target
    x = rng.normal(size=(200, 20))
    if signal:
        x[:, 0] += 3.0 * target
    train = np.arange(200) < 134
    return x, y, groups, train


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_a_real_signal_is_found_and_noise_is_not(task_type):
    scores = {}
    for signal in (True, False):
        x, y, groups, train = _signal_or_noise(task_type, signal)
        classes = np.unique(y) if task_type == "classification" else np.array([])
        pred = fit_predict(
            task_type, x[train], y[train], x[~train], classes, groups[train]
        )
        if task_type == "classification":
            scores[signal] = compute_auroc(y[~train], pred, classes)
        else:
            scores[signal] = compute_pearson(y[~train], pred)

    chance = 0.5 if task_type == "classification" else 0.0
    assert scores[True] > 0.9
    assert abs(scores[False] - chance) < 0.25


def test_a_class_missing_from_training_gets_a_zero_column():
    """Class 1 is missing: classes 0 and 2 keep their own probability columns."""
    rng = np.random.default_rng(4)
    groups = np.repeat([f"p{i:02d}" for i in range(20)], 2)
    y = np.repeat(np.arange(20) % 2 * 2, 2)
    x = rng.normal(size=(40, 5)) + y[:, None]

    pred = fit_predict(
        "classification", x, y, rng.normal(size=(6, 5)), np.array([0, 1, 2]), groups
    )

    assert pred.shape == (6, 3)
    assert np.all(pred[:, 1] == 0.0)
    assert np.allclose(pred.sum(axis=1), 1.0)


def test_a_single_class_training_fold_raises():
    groups = np.repeat([f"p{i}" for i in range(6)], 2)
    with pytest.raises(SubmissionError, match="single class"):
        fit_predict(
            "classification",
            np.ones((12, 3)),
            np.zeros(12, dtype=int),
            np.ones((2, 3)),
            np.array([0, 1]),
            groups,
        )


def test_a_class_with_one_subject_raises():
    groups = np.repeat([f"p{i}" for i in range(6)], 2)
    y = np.repeat([0, 0, 0, 0, 0, 1], 2)
    x = np.random.default_rng(5).normal(size=(12, 3))
    with pytest.raises(SubmissionError, match="fewer than two subjects in a class"):
        fit_predict("classification", x, y, x[:2], np.array([0, 1]), groups)


def test_regression_with_one_subject_raises():
    x = np.random.default_rng(6).normal(size=(4, 3))
    with pytest.raises(SubmissionError, match="fewer than two subjects"):
        fit_predict(
            "regression", x, np.arange(4.0), x[:1], np.array([]), np.array(["p0"] * 4)
        )
