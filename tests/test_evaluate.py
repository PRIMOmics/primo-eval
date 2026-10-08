import numpy as np
import pandas as pd
import pytest

from primomics import evaluate as ev
from primomics.errors import EvaluatorError, SubmissionError
from primomics.submission import DATASET_ID, SAMPLE_ID


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


def test_a_test_fold_with_one_class_is_our_error():
    """The folds are ours: a fold that cannot be scored must not cost the submitter."""
    y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.repeat([0, 1, 2], 4)
    emb = np.random.default_rng(8).normal(size=(12, 3))
    task = _task("t901", "d901", "classification", "auroc")
    with pytest.raises(EvaluatorError, match="CV fold 0 holds a single label value"):
        ev.score_task(task, _labels(y, folds), _emb_frame(emb))


def test_a_transfer_test_cohort_with_one_class_is_our_error():
    y = np.array([0, 1, 0, 1, 0, 0])
    split = np.array([ev.SPLIT_TRAIN] * 4 + [ev.SPLIT_TEST] * 2)
    labels = _transfer_labels(y, split)
    task = _task("t904", "d904", "classification", "auroc")
    with pytest.raises(EvaluatorError, match="transfer test cohort"):
        ev.score_task(
            task, labels, _emb_frame(np.random.default_rng(3).normal(size=(6, 3)))
        )


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


def test_transfer_single_class_train_is_our_error():
    y = np.array([0, 0, 0, 0, 1, 0])
    split = np.array([ev.SPLIT_TRAIN] * 4 + [ev.SPLIT_TEST] * 2)
    ids = [f"s{i}" for i in range(6)]
    emb = np.random.default_rng(0).normal(size=(6, 5))
    task = _task("t904", "d904", "classification", "auroc")
    with pytest.raises(EvaluatorError, match="single class"):
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


def _write_csv(path, blocks):
    """blocks = {dataset_id: (ids, matrix)} -> a multi-dataset CSV (NaN-padded)."""
    frames = []
    for dataset_id, (ids, mat) in blocks.items():
        df = pd.DataFrame(mat, columns=[f"e{j}" for j in range(mat.shape[1])])
        df.insert(0, SAMPLE_ID, ids)
        df.insert(0, DATASET_ID, dataset_id)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _no_fetch(task_id):
    raise AssertionError("fetch_labels should not have been called")


def test_scoreable_tasks_skips_orphans():
    tasks = [
        _task("t901", "d901", "classification", "auroc"),
        _task("t909", "d999", "regression", "pearson"),
    ]
    kept = ev.scoreable_tasks(tasks, {"d901"})
    assert [t["task_id"] for t in kept] == ["t901"]


def _two_task_submission(tmp_path, signal=True):
    _, cl, cids, cemb = _classification(signal=signal)
    _, rl, rids, remb = _regression(signal=signal)
    _write_csv(tmp_path / "sub.csv", {"d901": (cids, cemb), "d902": (rids, remb)})
    tasks = [
        _task("t901", "d901", "classification", "auroc"),
        _task("t902", "d902", "regression", "pearson"),
    ]
    labels_by = {"t901": cl, "t902": rl}
    datasets = [{"id": "d901"}, {"id": "d902"}]
    return tmp_path / "sub.csv", datasets, tasks, labels_by.__getitem__


def test_score_all_full_coverage_signal(tmp_path):
    path, datasets, tasks, fetch = _two_task_submission(tmp_path, signal=True)
    res = ev.score_all(path, datasets=datasets, tasks=tasks, fetch_labels=fetch)
    assert res["full_coverage"] is True
    assert res["n_scored"] == 2 and res["n_total"] == 2
    assert res["coverage"] == 1.0
    assert res["categories"]["treatment_outcome"]["mean"] > 0.8
    assert res["categories"]["clinical_scores"]["mean"] > 0.6
    assert set(res["categories"]) == {"treatment_outcome", "clinical_scores"}
    assert res["missing"] == [] and res["invalid"] == []
    assert res["incomplete"] == []
    assert res["n_datasets_scored"] == 2 and res["n_datasets_total"] == 2


def test_score_all_matches_the_live_evaluator(tmp_path):
    """The live evaluator's summary of this submission, read on the unit sphere."""
    cls_task, cls_labels, emb = _fixed_task("classification")
    reg_task, reg_labels, _ = _fixed_task("regression")
    _write_csv(tmp_path / "sub.csv", {"d901": (list(emb.index), emb.to_numpy())})
    tasks = [
        cls_task,
        {**reg_task, "dataset_id": "d901"},
        _task("t903", "d903", "regression", "pearson"),
    ]
    datasets = [
        {"id": "d901", "modality": "bulk RNA"},
        {"id": "d903", "modality": "bulk RNA"},
    ]
    labels_by = {"t901": cls_labels, "t902": reg_labels}

    res = ev.score_all(
        tmp_path / "sub.csv",
        modality="bulk-rna",
        datasets=datasets,
        tasks=tasks,
        fetch_labels=labels_by.__getitem__,
    )

    assert res["categories"]["treatment_outcome"]["mean"] == pytest.approx(
        0.9301706349206349, abs=1e-9
    )
    assert res["categories"]["clinical_scores"]["mean"] == pytest.approx(
        0.9118100571001598, abs=1e-9
    )
    assert res["missing"] == ["d903"]
    assert res["coverage"] == pytest.approx(2 / 3)
    assert (res["n_scored"], res["n_total"], res["full_coverage"]) == (2, 3, False)
    assert (res["n_datasets_scored"], res["n_datasets_total"]) == (1, 2)
    assert res["invalid"] == [] and res["incomplete"] == []


def test_score_all_scopes_submission_to_selected_modality(tmp_path):
    task, labels, ids, emb = _classification()
    _write_csv(tmp_path / "sub.csv", {"d901": (ids, emb)})
    tasks = [task, _task("t903", "d903", "regression", "pearson")]
    datasets = [
        {"id": "d901", "modality": "bulk RNA"},
        {"id": "d903", "modality": "Single Cell RNA"},
    ]
    result = ev.score_all(
        tmp_path / "sub.csv",
        modality="bulk-rna",
        datasets=datasets,
        tasks=tasks,
        fetch_labels=lambda task_id: labels,
    )
    assert result["n_datasets_total"] == 1
    assert result["full_coverage"] is True
    with pytest.raises(SubmissionError, match="outside the selected modality"):
        ev.score_all(
            tmp_path / "sub.csv",
            modality="single-cell-rna",
            datasets=datasets,
            tasks=tasks,
            fetch_labels=_no_fetch,
        )


def test_unknown_modality_is_the_submitters_error(tmp_path):
    path, datasets, tasks, fetch = _two_task_submission(tmp_path)
    with pytest.raises(SubmissionError, match="unknown or unavailable modality"):
        ev.score_all(
            path,
            modality="proteomics",
            datasets=datasets,
            tasks=tasks,
            fetch_labels=fetch,
        )


def test_score_all_null_is_chance(tmp_path):
    path, datasets, tasks, fetch = _two_task_submission(tmp_path, signal=False)
    res = ev.score_all(path, datasets=datasets, tasks=tasks, fetch_labels=fetch)
    assert res["categories"]["treatment_outcome"]["mean"] < 0.7


def test_multi_task_per_dataset(tmp_path):
    _, cl, cids, cemb = _classification(signal=True)
    yreg = np.random.default_rng(3).normal(size=len(cids))
    rl = _labels(yreg, _folds(len(cids)), cids)
    _write_csv(tmp_path / "s.csv", {"d901": (cids, cemb)})
    tasks = [
        _task("t901", "d901", "classification", "auroc"),
        _task("t902", "d901", "regression", "pearson"),
    ]
    labels_by = {"t901": cl, "t902": rl}
    res = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=labels_by.__getitem__,
    )
    assert res["n_total"] == 2 and res["n_scored"] == 2
    assert res["full_coverage"] is True
    assert res["n_datasets_total"] == 1 and res["n_datasets_scored"] == 1
    assert set(res["categories"]) == {"treatment_outcome", "clinical_scores"}
    assert res["incomplete"] == []


def test_missing_dataset_makes_all_its_tasks_missing(tmp_path):
    _, cl, cids, cemb = _classification()
    _write_csv(tmp_path / "s.csv", {"d901": (cids, cemb)})
    tasks = [
        _task("t901", "d901", "classification", "auroc"),
        _task("t902", "d902", "regression", "pearson"),
    ]
    res = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d901"}, {"id": "d902"}],
        tasks=tasks,
        fetch_labels={"t901": cl}.__getitem__,
    )
    assert res["missing"] == ["d902"]
    assert res["full_coverage"] is False
    assert res["n_scored"] == 1 and res["n_total"] == 2
    assert res["n_datasets_scored"] == 1
    assert res["incomplete"] == []


def test_invalid_task_when_dataset_embedded(tmp_path):
    _, cl, cids, cemb = _classification(n=60)
    _write_csv(tmp_path / "s.csv", {"d901": (cids, cemb)})
    tasks = [
        _task("t901", "d901", "classification", "auroc"),
        _task("t902", "d901", "classification", "auroc"),
    ]
    bad = _labels(
        np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3]), ids=["z0", "z1", "z2", "z3"]
    )
    res = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels={"t901": cl, "t902": bad}.__getitem__,
    )
    assert res["n_scored"] == 1 and res["full_coverage"] is False
    invalid = res["invalid"][0]
    assert invalid["task_id"] == "t902" and invalid["dataset_id"] == "d901"
    assert res["incomplete"] == ["d901"]
    assert res["n_datasets_scored"] == 0


def test_block_invalid_fails_all_its_tasks(tmp_path):
    df = pd.DataFrame(
        {
            DATASET_ID: ["d901", "d901"],
            SAMPLE_ID: ["a", "a"],
            "e0": [1.0, 2.0],
            "e1": [3.0, 4.0],
        }
    )
    df.to_csv(tmp_path / "s.csv", index=False)
    tasks = [
        _task("t901", "d901", "classification", "auroc"),
        _task("t902", "d901", "regression", "pearson"),
    ]
    res = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=_no_fetch,
    )
    assert res["n_scored"] == 0
    reasons = {i["task_id"]: i["reason"] for i in res["invalid"]}
    assert set(reasons) == {"t901", "t902"}
    assert "duplicate" in reasons["t901"] and reasons["t901"] == reasons["t902"]
    assert res["incomplete"] == ["d901"]


def test_score_all_unknown_dataset_raises(tmp_path):
    _, _, cids, cemb = _classification()
    _write_csv(
        tmp_path / "s.csv", {"d901": (cids, cemb), "d999": (["a"], np.ones((1, 8)))}
    )
    tasks = [_task("t901", "d901", "classification", "auroc")]
    with pytest.raises(SubmissionError, match="unknown"):
        ev.score_all(
            tmp_path / "s.csv",
            datasets=[{"id": "d901"}],
            tasks=tasks,
            fetch_labels=_no_fetch,
        )


def test_incomplete_dataset_is_invalid(tmp_path):
    _, cl, cids, cemb = _classification(n=60)
    _write_csv(tmp_path / "s.csv", {"d901": (cids[:-5], cemb[:-5])})
    tasks = [_task("t901", "d901", "classification", "auroc")]
    res = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=lambda task_id: cl,
    )
    assert res["n_scored"] == 0
    assert res["invalid"][0]["task_id"] == "t901"
    assert "missing" in res["invalid"][0]["reason"]
    assert res["incomplete"] == ["d901"]


def test_scoring_error_is_invalid(tmp_path):
    n = 20
    folds = np.repeat(np.arange(5), 4)
    y = np.tile([0, 1], n // 2)
    ids = [f"s{i}" for i in range(n)]
    emb = np.random.default_rng(0).normal(size=(n, 6))
    emb[3, 2] = np.nan
    labels = _labels(y, folds, ids)
    _write_csv(tmp_path / "s.csv", {"d901": (ids, emb)})
    tasks = [_task("t901", "d901", "classification", "auroc")]
    res = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=lambda task_id: labels,
    )
    assert res["n_scored"] == 0
    assert "NaN or inf" in res["invalid"][0]["reason"]


def test_a_degenerate_transfer_score_is_invalid(tmp_path):
    """Constant predictions give a NaN Pearson: the task is invalid, not scored."""
    y = np.random.default_rng(10).normal(size=20)
    labels = _transfer_labels(y, [ev.SPLIT_TRAIN] * 14 + [ev.SPLIT_TEST] * 6)
    ids = list(labels[SAMPLE_ID])
    _write_csv(tmp_path / "s.csv", {"d902": (ids, np.ones((20, 3)))})
    res = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d902"}],
        tasks=[_task("t902", "d902", "regression", "pearson")],
        fetch_labels=lambda task_id: labels,
        representation=ev.GENE_SPACE,
    )
    assert res["n_scored"] == 0
    assert res["invalid"][0]["reason"] == "degenerate score (constant predictions)"


def test_a_broken_fold_stops_the_scoring(tmp_path):
    """Our error is raised, never recorded as the submitter's invalid task."""
    y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    labels = _labels(y, np.repeat([0, 1, 2], 4))
    ids = list(labels[SAMPLE_ID])
    _write_csv(
        tmp_path / "s.csv",
        {"d901": (ids, np.random.default_rng(8).normal(size=(12, 3)))},
    )
    with pytest.raises(EvaluatorError, match="single label value"):
        ev.score_all(
            tmp_path / "s.csv",
            datasets=[{"id": "d901"}],
            tasks=[_task("t901", "d901", "classification", "auroc")],
            fetch_labels=lambda task_id: labels,
        )


def test_fetch_failure_raises_evaluator_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "FETCH_BACKOFF", 0.0)
    _, _, cids, cemb = _classification()
    _write_csv(tmp_path / "s.csv", {"d901": (cids, cemb)})
    tasks = [_task("t901", "d901", "classification", "auroc")]
    calls = []

    def boom(task_id):
        calls.append(task_id)
        raise ConnectionError("hub unreachable")

    with pytest.raises(EvaluatorError, match="could not load"):
        ev.score_all(
            tmp_path / "s.csv",
            datasets=[{"id": "d901"}],
            tasks=tasks,
            fetch_labels=boom,
        )
    assert calls == ["t901"] * ev.FETCH_ATTEMPTS


def test_a_flaky_fetch_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "FETCH_BACKOFF", 0.0)
    path, datasets, tasks, fetch = _two_task_submission(tmp_path)
    failures = {"t901": 2}

    def flaky(task_id):
        if failures.get(task_id, 0):
            failures[task_id] -= 1
            raise ConnectionError("hub hiccup")
        return fetch(task_id)

    res = ev.score_all(path, datasets=datasets, tasks=tasks, fetch_labels=flaky)
    assert res["full_coverage"] is True


def test_score_all_transfer_two_tasks(tmp_path):
    _, forward, ids, emb = _transfer_classification(signal=True)
    y = forward[ev.LABEL].to_numpy()
    reverse_split = np.where(
        forward[ev.SPLIT].to_numpy() == ev.SPLIT_TRAIN, ev.SPLIT_TEST, ev.SPLIT_TRAIN
    )
    backward = _transfer_labels(y, reverse_split, ids)
    _write_csv(tmp_path / "sub.csv", {"d904": (ids, emb)})
    tasks = [
        _task("t904", "d904", "classification", "auroc"),
        _task("t905", "d904", "classification", "auroc"),
    ]
    res = ev.score_all(
        tmp_path / "sub.csv",
        datasets=[{"id": "d904"}],
        tasks=tasks,
        fetch_labels={"t904": forward, "t905": backward}.__getitem__,
    )
    assert res["n_total"] == 2 and res["n_scored"] == 2
    assert res["full_coverage"] is True
    assert res["n_datasets_total"] == 1 and res["n_datasets_scored"] == 1
    assert set(res["categories"]) == {"treatment_outcome"}


def test_embedding_submission_is_read_on_the_unit_sphere(tmp_path):
    """Rescaling one patient's vector must not move an embedding score."""
    path, datasets, tasks, fetch = _two_task_submission(tmp_path, signal=True)
    frame = pd.read_csv(path)
    dims = [c for c in frame.columns if c not in {DATASET_ID, SAMPLE_ID}]
    frame.loc[::2, dims] = frame.loc[::2, dims] * 7.0
    frame.to_csv(tmp_path / "scaled.csv", index=False)

    plain = ev.score_all(path, datasets=datasets, tasks=tasks, fetch_labels=fetch)
    stretched = ev.score_all(
        tmp_path / "scaled.csv", datasets=datasets, tasks=tasks, fetch_labels=fetch
    )

    assert [round(t.score, 10) for t in plain["per_task"]] == [
        round(t.score, 10) for t in stretched["per_task"]
    ]


def test_gene_space_submission_keeps_its_own_scale(tmp_path):
    """The same rescaling does move a gene-space score: that scale is a measurement."""
    path, datasets, tasks, fetch = _two_task_submission(tmp_path, signal=True)
    frame = pd.read_csv(path)
    dims = [c for c in frame.columns if c not in {DATASET_ID, SAMPLE_ID}]
    frame.loc[::2, dims] = frame.loc[::2, dims] * 7.0
    frame.to_csv(tmp_path / "scaled.csv", index=False)

    kwargs = {"datasets": datasets, "tasks": tasks, "fetch_labels": fetch}
    as_genes = ev.score_all(
        tmp_path / "scaled.csv", representation=ev.GENE_SPACE, **kwargs
    )
    as_embedding = ev.score_all(
        tmp_path / "scaled.csv", representation=ev.EMBEDDING, **kwargs
    )

    assert [t.score for t in as_genes["per_task"]] != [
        t.score for t in as_embedding["per_task"]
    ]


def test_unknown_representation_is_the_submitters_error(tmp_path):
    path, datasets, tasks, fetch = _two_task_submission(tmp_path)
    with pytest.raises(SubmissionError, match="unknown representation"):
        ev.score_all(
            path,
            representation="whatever",
            datasets=datasets,
            tasks=tasks,
            fetch_labels=fetch,
        )
