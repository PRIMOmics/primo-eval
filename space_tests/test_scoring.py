"""Unit tests for the pure scoring policy (no network, no probe)."""

import numpy as np
import pytest
from scoring import (
    TaskScore,
    category_means,
    compute_auroc,
    compute_pearson,
    compute_residual_sample_spearman,
    compute_target_centered_sample_spearman,
    sort_key,
)


def _score(task_id, dataset_id, category, metric, score):
    return TaskScore(task_id, dataset_id, category, metric, score, 10)


def test_compute_auroc_binary():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
    assert compute_auroc(y, proba) == pytest.approx(1.0)


def test_transfer_auroc_reads_the_column_of_the_class_it_scores():
    """Test cohort holds 2 of 3 classes: scoring class 1's column would be wrong."""
    y_true = np.array([0, 0, 2, 2])
    preds = np.array(
        [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.3, 0.6], [0.2, 0.2, 0.6]]
    )

    assert compute_auroc(y_true, preds, np.array([0, 1, 2])) == 1.0
    assert compute_auroc(y_true, preds[:, ::-1], np.array([0, 1, 2])) == 0.0


def test_auroc_survives_a_class_the_training_fold_never_saw():
    """Zeroed column + renormalisation, instead of sklearn's sum-to-one refusal."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    preds = np.zeros((6, 3))
    preds[:, 0] = [0.9, 0.8, 0.2, 0.1, 0.2, 0.1]
    preds[:, 1] = [0.1, 0.2, 0.8, 0.9, 0.3, 0.2]

    score = compute_auroc(y_true, preds, np.array([0, 1, 2]))

    assert 0.0 <= score <= 1.0


def test_auroc_refuses_predictions_with_no_mass_on_the_present_classes():
    y_true = np.array([0, 0, 1, 1])
    preds = np.zeros((4, 3))
    preds[:, 2] = 1.0

    with pytest.raises(ValueError, match="no probability"):
        compute_auroc(y_true, preds, np.array([0, 1, 2]))


def test_compute_pearson_and_degenerate():
    perfect = compute_pearson(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))
    assert perfect == pytest.approx(1.0)
    assert np.isnan(compute_pearson(np.array([1.0, 2, 3]), np.array([5.0, 5, 5])))


def test_residual_sample_spearman_anchors_mean_response_at_zero():
    training_mean = np.array([10.0, 20.0, 30.0, 40.0])
    truth = training_mean + np.array([[1.0, 3.0, -1.0, 2.0]])
    baseline = training_mean[None, :]

    assert compute_residual_sample_spearman(
        truth, truth, training_mean
    ) == pytest.approx(1.0)
    assert compute_residual_sample_spearman(
        truth, baseline, training_mean
    ) == pytest.approx(0.0)


def test_target_centered_spearman_removes_shared_human_response():
    truth = np.array([[1.0, 4.0, 2.0], [3.0, 2.0, 5.0]])
    shared_prediction = np.array([[2.0, 8.0, 4.0], [2.0, 8.0, 4.0]])

    assert compute_target_centered_sample_spearman(
        truth, shared_prediction
    ) == pytest.approx(0.0)
    assert compute_target_centered_sample_spearman(truth, truth) == pytest.approx(1.0)


def test_category_means_groups_by_category():
    scores = [
        _score("t1", "d1", "treatment_outcome", "auroc", 0.8),
        _score("t2", "d2", "treatment_outcome", "auroc", 0.6),
        _score("t3", "d3", "clinical_scores", "pearson", 0.4),
    ]
    cats = category_means(scores)
    assert cats["treatment_outcome"] == {
        "metric": "auroc",
        "mean": pytest.approx(0.7),
        "n_tasks": 2,
    }
    assert cats["clinical_scores"]["mean"] == pytest.approx(0.4)
    assert cats["clinical_scores"]["n_tasks"] == 1


def test_category_means_drops_non_finite_from_mean():
    scores = [
        _score("t1", "d1", "clinical_scores", "pearson", 0.4),
        _score("t2", "d2", "clinical_scores", "pearson", float("nan")),
    ]
    cats = category_means(scores)
    assert cats["clinical_scores"]["mean"] == pytest.approx(0.4)
    assert cats["clinical_scores"]["n_tasks"] == 2


def test_category_means_all_non_finite_is_nan():
    scores = [_score("t1", "d1", "endotype", "auroc", float("nan"))]
    assert np.isnan(category_means(scores)["endotype"]["mean"])


def test_category_means_empty():
    assert category_means([]) == {}


def test_sort_key_is_mean_of_category_means():
    cats = {
        "treatment_outcome": {"metric": "auroc", "mean": 0.8, "n_tasks": 1},
        "clinical_scores": {"metric": "pearson", "mean": 0.2, "n_tasks": 1},
    }
    assert sort_key(cats) == pytest.approx(0.5)


def test_sort_key_ignores_non_finite_and_empty():
    cats = {
        "treatment_outcome": {"metric": "auroc", "mean": 0.8, "n_tasks": 1},
        "endotype": {"metric": "auroc", "mean": float("nan"), "n_tasks": 1},
    }
    assert sort_key(cats) == pytest.approx(0.8)


def test_nothing_finite_sorts_last_instead_of_at_zero():
    """A constant embedding scores NaN everywhere; that must not beat a bad score."""
    nothing = {"clinical_scores": {"metric": "pearson", "mean": float("nan")}}
    negative = {"clinical_scores": {"metric": "pearson", "mean": -0.2}}
    assert sort_key(nothing) == float("-inf")
    assert sort_key({}) == float("-inf")
    assert sort_key(nothing) < sort_key(negative)
