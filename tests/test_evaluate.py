import numpy as np
import pandas as pd
import pytest

from primomics import evaluate as ev
from primomics.errors import EvaluatorError, SubmissionError
from primomics.submission import SAMPLE_ID


def _labels(y, folds, ids=None, repeat_folds=None, subjects=None):
    ids = ids if ids is not None else [f"s{i}" for i in range(len(y))]
    subjects = subjects if subjects is not None else [f"p{i}" for i in range(len(y))]
    labels = pd.DataFrame(
        {SAMPLE_ID: ids, ev.SUBJECT_ID: subjects, ev.LABEL: y, ev.FOLD: folds}
    )
    if repeat_folds is not None:
        for repeat, values in enumerate(repeat_folds, start=1):
            labels[f"{ev.FOLD}_{repeat}"] = values
    return labels


def _emb_frame(matrix, ids=None):
    ids = ids if ids is not None else [f"s{i}" for i in range(len(matrix))]
    df = pd.DataFrame(matrix)
    df.index = ids
    return df


def _folds(n, k=5):
    return np.array([i % k for i in range(n)])


def _task(task_id, dataset_id, task_type, metric, category=None):
    if category is None:
        category = "treatment_outcome" if metric == "auroc" else "clinical_scores"
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "task_type": task_type,
        "metric": metric,
        "category": category,
    }


def _classification(n=60, seed=42, signal=True):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    emb = rng.normal(size=(n, 8))
    if signal:
        emb[:, 0] += 3.0 * y
    ids = [f"s{i}" for i in range(n)]
    task = _task("t901", "d901", "classification", "auroc")
    return task, _labels(y, _folds(n), ids), ids, emb


def _regression(n=60, seed=7, signal=True):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    emb = rng.normal(size=(n, 8))
    if signal:
        emb[:, 0] += 2.0 * y
    ids = [f"r{i}" for i in range(n)]
    task = _task("t902", "d902", "regression", "pearson")
    return task, _labels(y, _folds(n), ids), ids, emb


def _transfer_labels(y, split, ids=None):
    ids = ids if ids is not None else [f"s{i}" for i in range(len(y))]
    return pd.DataFrame(
        {
            SAMPLE_ID: ids,
            ev.SUBJECT_ID: [f"p{i}" for i in range(len(y))],
            ev.LABEL: y,
            ev.SPLIT: split,
        }
    )


def _transfer_classification(n_train=40, n_test=20, seed=42, signal=True):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = np.array([0, 1] * (n // 2))
    emb = rng.normal(size=(n, 8))
    if signal:
        emb[:, 0] += 3.0 * y
    split = np.array([ev.SPLIT_TRAIN] * n_train + [ev.SPLIT_TEST] * n_test)
    ids = [f"s{i}" for i in range(n)]
    task = _task("t904", "d904", "classification", "auroc")
    return task, _transfer_labels(y, split, ids), ids, emb


def test_repeated_cv_scores_score_every_external_partition(monkeypatch):
    labels = _labels(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        repeat_folds=[np.array([1, 0, 1, 0]), np.array([2, 0, 2, 0])],
    )
    seen = []

    def fold_score(metric, task_type, y, folds, matrix, classes, groups):
        seen.append(folds[0])
        return float(folds[0])

    monkeypatch.setattr(ev, "_fold_score", fold_score)
    monkeypatch.setattr(ev, "OUTER_CV_JOBS", 1)
    scores = ev._repeated_cv_scores(
        metric="auroc",
        task_type="classification",
        y=labels[ev.LABEL].to_numpy(),
        labels=labels,
        matrix=np.ones((4, 2)),
        classes=np.array([0, 1]),
    )
    assert seen == [0, 1, 2]
    assert scores == (0.0, 1.0, 2.0)


def test_classification_signal_and_null():
    task, labels, ids, emb = _classification(signal=True)
    res = ev.score_task(task, labels, _emb_frame(emb, ids))
    assert res.score > 0.9
    _, _, ids0, null = _classification(signal=False)
    res0 = ev.score_task(task, labels, _emb_frame(null, ids0))
    assert 0.3 < res0.score < 0.7


def test_regression_signal_and_null():
    task, labels, ids, emb = _regression(signal=True)
    res = ev.score_task(task, labels, _emb_frame(emb, ids))
    assert res.score > 0.8
    _, _, ids0, null = _regression(signal=False)
    res0 = ev.score_task(task, labels, _emb_frame(null, ids0))
    assert abs(res0.score) < 0.4


def test_ordinal_severity_scores_as_regression():
    rng = np.random.default_rng(25)
    y = np.tile(np.arange(6), 10)
    ids = [f"s{i:04d}" for i in range(len(y))]
    emb = rng.normal(size=(len(y), 8))
    emb[:, 0] += 2 * y
    task = _task("t903", "d903", "regression", "pearson")
    result = ev.score_task(task, _labels(y, _folds(len(y)), ids), _emb_frame(emb, ids))
    assert result.n_samples == len(ids)
    assert result.score > 0.9


def test_score_carries_ids_category_and_size():
    task, labels, ids, emb = _classification(n=30)
    res = ev.score_task(task, labels, _emb_frame(emb, ids))
    assert res.task_id == "t901"
    assert res.dataset_id == "d901"
    assert res.n_samples == 30
    assert res.category == "treatment_outcome"
    assert res.metric == "auroc"
    assert np.isfinite(res.score)


def _fixed_task(kind):
    """30 patients with 2 samples each, built from formulas: the same everywhere."""
    rows = np.arange(60)
    subject = rows // 2
    latent = np.sin(0.9 * subject)
    x = np.outer(latent, np.linspace(1.0, 0.2, 8)) + 0.8 * np.sin(
        np.outer(rows + 1, np.arange(1, 9))
    )
    ids = [f"s{i:02d}" for i in rows]
    labels = pd.DataFrame(
        {SAMPLE_ID: ids, ev.SUBJECT_ID: [f"p{s:02d}" for s in subject]}
    )
    if kind == "regression":
        labels[ev.LABEL] = latent + 0.3 * np.cos(2.0 * rows)
        task = _task("t902", "d902", "regression", "pearson")
    else:
        labels[ev.LABEL] = (latent > 0).astype(int)
        task = _task("t901", "d901", "classification", "auroc")
    if kind == "transfer":
        labels[ev.SPLIT] = np.where(subject < 20, ev.SPLIT_TRAIN, ev.SPLIT_TEST)
    else:
        for repeat in range(10):
            rank = np.argsort(np.argsort(np.sin((repeat + 1.3) * np.arange(30))))
            column = ev.FOLD if repeat == 0 else f"{ev.FOLD}_{repeat}"
            labels[column] = (rank % 3)[subject]
    return task, labels, _emb_frame(x, ids)


@pytest.mark.parametrize(
    ("kind", "expected", "n_repeats", "n_samples"),
    [
        ("classification", 0.9453670634920636, 10, 60),
        ("regression", 0.9407686709886036, 10, 60),
        ("transfer", 0.9791666666666666, 1, 20),
    ],
)
def test_matches_the_live_evaluator_on_fixed_tasks(
    kind, expected, n_repeats, n_samples
):
    """Scores the live evaluator gives on these tasks: a changed protocol moves them."""
    task, labels, emb = _fixed_task(kind)

    result = ev.score_task(task, labels, emb)

    assert result.score == pytest.approx(expected, abs=1e-9)
    assert len(result.repeat_scores) == n_repeats
    assert result.n_samples == n_samples


def test_parallel_and_sequential_repeats_agree(monkeypatch):
    task, labels, emb = _fixed_task("classification")
    parallel = ev.score_task(task, labels, emb)
    monkeypatch.setattr(ev, "OUTER_CV_JOBS", 1)
    sequential = ev.score_task(task, labels, emb)

    assert sequential.repeat_scores == pytest.approx(parallel.repeat_scores, abs=1e-12)


def test_missing_sample_raises():
    emb = _emb_frame(np.random.default_rng(1).normal(size=(5, 3)))
    labels = _labels(np.array([0, 1, 0, 1, 0, 1]), _folds(6))
    with pytest.raises(SubmissionError, match="missing"):
        ev._align(labels, emb)


def test_nan_raises():
    matrix = np.ones((4, 3))
    matrix[0, 0] = np.nan
    labels = _labels(np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3]))
    with pytest.raises(SubmissionError, match="NaN"):
        ev._align(labels, _emb_frame(matrix))


def test_id_normalisation_int_vs_str():
    emb = _emb_frame(np.eye(4), ids=["0", "1", "2", "3"])
    labels = _labels(np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3]), ids=[0, 1, 2, 3])
    assert ev._align(labels, emb).shape == (4, 4)


def test_rows_are_matched_by_sample_id_not_by_position():
    labels = _labels(np.array([0, 1, 0]), np.array([0, 1, 2]), ids=["b", "c", "a"])
    emb = _emb_frame(np.array([[1.0], [2.0], [3.0]]), ids=["a", "b", "c"])
    assert ev._align(labels, emb).ravel().tolist() == [2.0, 3.0, 1.0]


def test_unknown_metric_raises_evaluator_error():
    task, labels, ids, emb = _classification()
    task["metric"] = "bogus"
    with pytest.raises(EvaluatorError, match="unknown metric"):
        ev.score_task(task, labels, _emb_frame(emb, ids))


def test_a_test_fold_with_one_class_is_the_submitters_error():
    """AUROC on one class is NaN in scikit-learn, so the fold has no score."""
    y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.repeat([0, 1, 2], 4)
    emb = np.random.default_rng(8).normal(size=(12, 3))
    task = _task("t901", "d901", "classification", "auroc")
    with pytest.raises(SubmissionError, match="non-finite"):
        ev.score_task(task, _labels(y, folds), _emb_frame(emb))


def test_constant_predictions_are_the_submitters_error():
    """A constant embedding gives a constant ridge prediction: Pearson is NaN."""
    y = np.random.default_rng(9).normal(size=30)
    task = _task("t902", "d902", "regression", "pearson")
    with pytest.raises(SubmissionError, match="non-finite"):
        ev.score_task(task, _labels(y, _folds(30, 3)), _emb_frame(np.ones((30, 4))))


def test_transfer_scoring_signal_and_null():
    task, labels, ids, emb = _transfer_classification(signal=True)
    res = ev.score_task(task, labels, _emb_frame(emb, ids))
    assert res.score > 0.8
    assert res.n_samples == 20
    task0, labels0, ids0, null = _transfer_classification(signal=False)
    res0 = ev.score_task(task0, labels0, _emb_frame(null, ids0))
    assert 0.3 < res0.score < 0.7


def test_transfer_single_class_train_raises_submission_error():
    y = np.array([0, 0, 0, 0, 1, 0])
    split = np.array([ev.SPLIT_TRAIN] * 4 + [ev.SPLIT_TEST] * 2)
    ids = [f"s{i}" for i in range(6)]
    emb = np.random.default_rng(0).normal(size=(6, 5))
    task = _task("t904", "d904", "classification", "auroc")
    with pytest.raises(SubmissionError):
        ev.score_task(task, _transfer_labels(y, split, ids), _emb_frame(emb, ids))


def test_transfer_with_an_empty_side_is_our_error():
    y = np.array([0, 1, 0, 1])
    labels = _transfer_labels(y, [ev.SPLIT_TRAIN] * 4)
    task = _task("t904", "d904", "classification", "auroc")
    with pytest.raises(EvaluatorError, match="empty train or test side"):
        ev.score_task(task, labels, _emb_frame(np.eye(4)))


def test_load_labels_accepts_repeated_fold_or_split(tmp_path):
    _labels(
        np.array([0, 1, 0, 1]),
        _folds(4),
        repeat_folds=[np.array([1, 0, 1, 0]), np.array([2, 0, 2, 0])],
    ).to_csv(tmp_path / "f.csv", index=False)
    labels = ev.load_labels(tmp_path / "f.csv")
    assert ev._fold_columns(labels) == ["fold", "fold_1", "fold_2"]
    _transfer_labels(np.array([0, 1]), ["train", "test"]).to_csv(
        tmp_path / "s.csv", index=False
    )
    assert ev.SPLIT in ev.load_labels(tmp_path / "s.csv").columns


def test_load_labels_rejects_both_and_neither(tmp_path):
    both = pd.DataFrame(
        {
            SAMPLE_ID: ["a"],
            ev.SUBJECT_ID: ["p1"],
            ev.LABEL: [0],
            ev.FOLD: [0],
            ev.SPLIT: ["train"],
        }
    )
    both.to_csv(tmp_path / "both.csv", index=False)
    with pytest.raises(EvaluatorError, match="exactly one"):
        ev.load_labels(tmp_path / "both.csv")
    neither = pd.DataFrame({SAMPLE_ID: ["a"], ev.SUBJECT_ID: ["p1"], ev.LABEL: [0]})
    neither.to_csv(tmp_path / "neither.csv", index=False)
    with pytest.raises(EvaluatorError, match="exactly one"):
        ev.load_labels(tmp_path / "neither.csv")


def test_load_labels_requires_subject_ids(tmp_path):
    pd.DataFrame({SAMPLE_ID: ["s1", "s2"], ev.LABEL: [0, 1], ev.FOLD: [0, 1]}).to_csv(
        tmp_path / "labels.csv", index=False
    )

    with pytest.raises(EvaluatorError, match="subject_id"):
        ev.load_labels(tmp_path / "labels.csv")


@pytest.mark.parametrize(
    ("subjects", "message"),
    [([None, "p1"], "missing subject_id"), ([" ", "p1"], "empty subject_id")],
)
def test_unusable_subject_ids_are_our_error(subjects, message):
    labels = _labels(np.array([0, 1]), np.array([0, 1]), subjects=subjects)
    with pytest.raises(EvaluatorError, match=message):
        ev._subject_groups(labels)


def test_a_badly_named_repeat_fold_column_is_our_error():
    labels = _labels(np.array([0, 1]), np.array([0, 1]))
    labels["fold_x"] = [0, 1]
    with pytest.raises(EvaluatorError, match="invalid repeat-fold columns"):
        ev._fold_columns(labels)
