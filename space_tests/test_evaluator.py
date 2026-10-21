"""Unit tests for the standalone PRIMO probe + score_all (no network)."""

import io
import pickle

import evaluator as ev
import huggingface_hub
import numpy as np
import pandas as pd
import pytest
from huggingface_hub.utils import EntryNotFoundError
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score


def test_all_repos_live_under_the_primo_org():
    assert ev.ORG == "PRIMOmics"
    assert ev.PUBLIC_REPO == "PRIMOmics/primo"
    assert ev.LABELS_REPO == "PRIMOmics/primo-labels"
    assert ev.RESULTS_REPO == "PRIMOmics/primo-results"


def _labels(y, folds, ids=None, repeat_folds=None, subjects=None):
    ids = ids if ids is not None else [f"s{i}" for i in range(len(y))]
    subjects = subjects if subjects is not None else [f"p{i}" for i in range(len(y))]
    labels = pd.DataFrame(
        {
            ev.SAMPLE_ID: ids,
            ev.SUBJECT_ID: subjects,
            ev.LABEL: y,
            ev.FOLD: folds,
        }
    )
    if repeat_folds is not None:
        for repeat, values in enumerate(repeat_folds, start=1):
            labels[f"{ev.FOLD}_{repeat}"] = values
    if subjects is not None:
        labels[ev.SUBJECT_ID] = subjects
    return labels


def _emb_frame(matrix, ids=None):
    ids = ids if ids is not None else [f"s{i}" for i in range(len(matrix))]
    df = pd.DataFrame(matrix)
    df.index = ids
    return df


def _folds(n, k=5):
    return np.array([i % k for i in range(n)])


def _task(task_id, dataset_id, task_type, metric, therapeutic_area, category=None):
    if category is None:
        category = "treatment_outcome" if metric == "auroc" else "clinical_scores"
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "task_type": task_type,
        "metric": metric,
        "category": category,
        "therapeutic_area": therapeutic_area,
        "disease": "PRIVATE",
        "tissue": "PRIVATE",
    }


def _classification(n=60, seed=42, signal=True):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    emb = rng.normal(size=(n, 8))
    if signal:
        emb[:, 0] += 3.0 * y
    ids = [f"s{i}" for i in range(n)]
    task = _task("t901", "d901", "classification", "auroc", "Gastro")
    return task, _labels(y, _folds(n), ids), ids, emb


def _regression(n=60, seed=7, signal=True):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    emb = rng.normal(size=(n, 8))
    if signal:
        emb[:, 0] += 2.0 * y
    ids = [f"r{i}" for i in range(n)]
    task = _task("t902", "d902", "regression", "pearson", "Derm")
    return task, _labels(y, _folds(n), ids), ids, emb


def _write_csv(path, blocks):
    """blocks = {dataset_id: (ids, matrix)} -> a multi-dataset CSV (NaN-padded)."""
    frames = []
    for dataset_id, (ids, mat) in blocks.items():
        df = pd.DataFrame(mat, columns=[f"e{j}" for j in range(mat.shape[1])])
        df.insert(0, ev.SAMPLE_ID, ids)
        df.insert(0, ev.DATASET_ID, dataset_id)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _no_fetch(task_id, token=None):
    raise AssertionError("fetch_labels should not have been called")


# --- probe: repeated fold-wise scoring ---


def test_repeated_cv_score_averages_external_partitions(monkeypatch):
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
    score = ev._repeated_cv_score(
        metric="auroc",
        task_type="classification",
        y=labels[ev.LABEL].to_numpy(),
        labels=labels,
        matrix=np.ones((4, 2)),
        classes=np.array([0, 1]),
    )
    assert seen == [0, 1, 2]
    assert score == 1.0


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


def test_ordinal_severity_uses_unchanged_sample_submission_schema():
    rng = np.random.default_rng(25)
    y = np.tile(np.arange(6), 10)
    ids = [f"s{i:04d}" for i in range(len(y))]
    emb = rng.normal(size=(len(y), 8))
    emb[:, 0] += 2 * y
    task = _task("t925", "d917", "regression", "pearson", "Infectious diseases")
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


def test_missing_sample_raises():
    emb = _emb_frame(np.random.default_rng(1).normal(size=(5, 3)))
    labels = _labels(np.array([0, 1, 0, 1, 0, 1]), _folds(6))
    with pytest.raises(ev.SubmissionError, match="missing"):
        ev._align(labels, emb)


def test_nan_raises():
    matrix = np.ones((4, 3))
    matrix[0, 0] = np.nan
    labels = _labels(np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3]))
    with pytest.raises(ev.SubmissionError, match="NaN"):
        ev._align(labels, _emb_frame(matrix))


def test_id_normalisation_int_vs_str():
    emb = _emb_frame(np.eye(4), ids=["0", "1", "2", "3"])
    labels = _labels(np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3]), ids=[0, 1, 2, 3])
    assert ev._align(labels, emb).shape == (4, 4)


# --- probe: leave-pair-out on small binary tasks ---


def _fixed_probe(scores, seen):
    """A probe that ignores training and returns a fixed score per row."""

    def fit_predict(
        task_type, x_train, y_train, x_test, classes, groups_train, inner_cv_jobs
    ):
        seen.append((x_train[:, 0].astype(int), x_test[:, 0].astype(int)))
        positive = scores[x_test[:, 0].astype(int)]
        return np.column_stack([1 - positive, positive])

    return fit_predict


def _use_fixed_probe(monkeypatch, scores, seen):
    monkeypatch.setattr(ev, "OUTER_CV_JOBS", 1)
    monkeypatch.setattr(ev, "fit_predict", _fixed_probe(scores, seen))


def _small_cohort(n_subjects, samples_per_subject):
    rng = np.random.default_rng(0)
    subjects = np.repeat([f"p{i}" for i in range(n_subjects)], samples_per_subject)
    y = np.repeat(rng.permutation(np.arange(n_subjects) % 2), samples_per_subject)
    labels = _labels(y, _folds(len(y), 3), subjects=subjects)
    return labels, np.arange(len(y))[:, None].astype(float), rng.random(len(y))


def _lpo(labels, matrix):
    y = labels[ev.LABEL].to_numpy()
    subjects = ev._subject_groups(labels)
    return ev._leave_pair_out_score(y, subjects, matrix, np.array([0, 1]))


def _protocol(monkeypatch, task, labels):
    monkeypatch.setattr(ev, "_leave_pair_out_score", lambda *_: 0.25)
    monkeypatch.setattr(ev, "_repeated_cv_scores", lambda **_: (0.75,) * 10)
    result = ev.score_task(task, labels, _emb_frame(np.ones((len(labels), 2))))
    return "lpo" if result.repeat_scores == (0.25,) else "cv"


@pytest.mark.parametrize("samples_per_subject", [1, 3])
def test_leave_pair_out_with_a_fixed_probe_equals_the_auroc(
    monkeypatch, samples_per_subject
):
    labels, matrix, scores = _small_cohort(10, samples_per_subject)
    _use_fixed_probe(monkeypatch, scores, [])
    score = _lpo(labels, matrix)
    assert score == pytest.approx(roc_auc_score(labels[ev.LABEL], scores))


def test_leave_pair_out_never_trains_on_a_held_out_subject(monkeypatch):
    labels, matrix, scores = _small_cohort(8, 2)
    seen = []
    _use_fixed_probe(monkeypatch, scores, seen)
    _lpo(labels, matrix)
    subjects = labels[ev.SUBJECT_ID].to_numpy()
    assert len(seen) == 4 * 4
    for train, test in seen:
        assert len(np.unique(subjects[test])) == 2
        assert set(subjects[train]).isdisjoint(subjects[test])


def test_parallel_and_sequential_leave_pair_out_agree(monkeypatch):
    labels, _, _ = _small_cohort(8, 2)
    matrix = np.random.default_rng(3).normal(size=(len(labels), 4))
    monkeypatch.setattr(ev, "OUTER_CV_JOBS", 2)
    parallel = _lpo(labels, matrix)
    monkeypatch.setattr(ev, "OUTER_CV_JOBS", 1)
    assert _lpo(labels, matrix) == parallel


def test_small_binary_task_with_signal_scores_one_split():
    y = np.arange(8) % 2
    emb = np.random.default_rng(0).normal(size=(8, 2))
    emb[:, 0] += 3.0 * y
    labels = _labels(y, _folds(8, 3), subjects=[f"p{i}" for i in range(8)])
    task = _task("t1", "d901", "classification", "auroc", "Gastro")
    result = ev.score_task(task, labels, _emb_frame(emb))
    assert result.repeat_scores == (result.score,)
    assert result.score > 0.8


@pytest.mark.parametrize(
    ("n_subjects", "samples_per_subject", "expected"),
    [(29, 1, "lpo"), (30, 1, "cv"), (5, 8, "lpo")],
)
def test_binary_tasks_under_30_subjects_use_leave_pair_out(
    monkeypatch, n_subjects, samples_per_subject, expected
):
    labels, _, _ = _small_cohort(n_subjects, samples_per_subject)
    task = _task("t1", "d901", "classification", "auroc", "Gastro")
    assert _protocol(monkeypatch, task, labels) == expected


def test_the_fold_draw_does_not_decide(monkeypatch):
    uneven = np.repeat([0, 1, 2], [13, 10, 9])
    labels = _labels(np.arange(32) % 2, _folds(32, 3), repeat_folds=[uneven])
    task = _task("t1", "d901", "classification", "auroc", "Gastro")
    assert _protocol(monkeypatch, task, labels) == "cv"


def test_paired_binary_tasks_keep_cv(monkeypatch):
    subjects = np.repeat([f"p{i}" for i in range(10)], 2)
    labels = _labels(np.tile([0, 1], 10), _folds(20, 3), subjects=subjects)
    task = _task("t1", "d901", "classification", "auroc", "Gastro")
    assert _protocol(monkeypatch, task, labels) == "cv"


@pytest.mark.parametrize(
    ("task_type", "metric", "y"),
    [
        ("classification", "auroc", np.arange(12) % 3),
        ("regression", "pearson", np.linspace(0.0, 1.0, 12)),
    ],
)
def test_small_multiclass_and_regression_tasks_keep_cv(
    monkeypatch, task_type, metric, y
):
    task = _task("t1", "d901", task_type, metric, "Gastro")
    assert _protocol(monkeypatch, task, _labels(y, _folds(12, 3))) == "cv"


def test_a_non_finite_leave_pair_out_score_is_invalid(monkeypatch):
    labels, matrix, scores = _small_cohort(4, 1)
    scores[:] = np.nan
    _use_fixed_probe(monkeypatch, scores, [])
    with pytest.raises(ev.SubmissionError, match="non-finite"):
        _lpo(labels, matrix)


def test_leave_pair_out_needs_one_label_per_subject():
    labels = _labels(
        np.array([0, 1, 1, 0]), _folds(4, 2), subjects=["p0", "p0", "p1", "p2"]
    )
    with pytest.raises(ev.EvaluatorError, match="more than one label"):
        _lpo(labels, np.ones((4, 2)))


def test_a_small_binary_task_without_subjects_is_our_error():
    labels = _labels(np.arange(12) % 2, _folds(12, 3)).drop(columns=ev.SUBJECT_ID)
    task = _task("t1", "d901", "classification", "auroc", "Gastro")
    with pytest.raises(ev.EvaluatorError, match="subject_id"):
        ev.score_task(task, labels, _emb_frame(np.ones((12, 2))))


# --- multi-dataset loader ---


def test_multidataset_loader_csv(tmp_path):
    rng = np.random.default_rng(0)
    blocks = {
        "d901": (["a", "b"], rng.normal(size=(2, 4))),
        "d902": (["0", "1", "2"], rng.normal(size=(3, 4))),
    }
    _write_csv(tmp_path / "sub.csv", blocks)
    out = ev.load_submission(tmp_path / "sub.csv")
    assert set(out) == {"d901", "d902"}
    assert ev._to_embedding_frame(out["d901"]).shape == (2, 4)
    assert list(ev._to_embedding_frame(out["d902"]).index) == ["0", "1", "2"]


def test_multidataset_loader_npz(tmp_path):
    rng = np.random.default_rng(1)
    ids = np.array(["a", "b", "c", "d"])
    dsids = np.array(["d901", "d901", "d902", "d902"])
    emb = rng.normal(size=(4, 5))
    np.savez(tmp_path / "sub.npz", dataset_ids=dsids, sample_ids=ids, embeddings=emb)
    out = ev.load_submission(tmp_path / "sub.npz")
    assert set(out) == {"d901", "d902"}
    assert ev._to_embedding_frame(out["d901"]).shape == (2, 5)


def test_npz_with_object_arrays_is_refused(tmp_path):
    """Object arrays need pickle, which can run code hidden in an uploaded file."""
    np.savez(
        tmp_path / "sub.npz",
        dataset_ids=np.array(["d901"], dtype=object),
        sample_ids=np.array(["a"], dtype=object),
        embeddings=np.ones((1, 2)),
    )
    with pytest.raises(ev.SubmissionError, match="Save ids as text"):
        ev.load_submission(tmp_path / "sub.npz")


def _saved(save, array):
    buffer = io.BytesIO()
    save(buffer, array)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        b"dataset_id,sample_id,e0\nd901,a,1.0\n",
        pickle.dumps({"sample_ids": ["a"]}),
        _saved(np.save, np.ones((2, 2))),
        _saved(np.savez, np.ones(3))[:60],
        b"",
    ],
    ids=["csv", "pickle", "npy", "truncated-npz", "empty"],
)
def test_a_file_that_is_not_an_npz_archive_is_refused(tmp_path, content):
    path = tmp_path / "sub.npz"
    path.write_bytes(content)
    with pytest.raises(ev.SubmissionError, match="not a .npz archive"):
        ev.load_submission(path)


def test_npz_with_a_corrupt_member_is_refused(tmp_path):
    """A byte-corrupted upload is the submitter's file, not an evaluator bug."""
    path = tmp_path / "sub.npz"
    np.savez(
        path,
        dataset_ids=np.array(["d901", "d901"]),
        sample_ids=np.array(["a", "b"]),
        embeddings=np.ones((2, 3)),
    )
    raw = bytearray(path.read_bytes())
    offset = raw.find(b"embeddings.npy") + len("embeddings.npy") + 140
    raw[offset] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ev.SubmissionError, match="cannot read the .npz arrays"):
        ev.load_submission(path)


def test_missing_dataset_id_column_raises(tmp_path):
    df = pd.DataFrame({ev.SAMPLE_ID: ["a", "b"], "e0": [0.1, 0.2]})
    df.to_csv(tmp_path / "sub.csv", index=False)
    with pytest.raises(ev.SubmissionError, match="dataset_id"):
        ev.load_submission(tmp_path / "sub.csv")


def test_ragged_padding_dims(tmp_path):
    blocks = {
        "d901": (["a", "b"], np.ones((2, 4))),
        "d902": (["a", "b"], np.ones((2, 6))),
    }
    _write_csv(tmp_path / "sub.csv", blocks)
    out = ev.load_submission(tmp_path / "sub.csv")
    assert ev._to_embedding_frame(out["d901"]).shape[1] == 4
    assert ev._to_embedding_frame(out["d902"]).shape[1] == 6


def test_sample_id_reused_across_datasets_ok(tmp_path):
    blocks = {
        "d901": (["x", "y"], np.ones((2, 3))),
        "d902": (["x", "y"], np.ones((2, 3))),
    }
    _write_csv(tmp_path / "sub.csv", blocks)
    out = ev.load_submission(tmp_path / "sub.csv")
    assert ev._to_embedding_frame(out["d901"]).shape == (2, 3)
    assert ev._to_embedding_frame(out["d902"]).shape == (2, 3)


def test_duplicate_within_dataset_raises():
    df = pd.DataFrame({ev.SAMPLE_ID: ["a", "a"], "e0": [1.0, 2.0], "e1": [3.0, 4.0]})
    with pytest.raises(ev.SubmissionError, match="duplicate"):
        ev._to_embedding_frame(df)


# --- registry ---


def test_load_tasks_registry_duplicate_raises(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "tasks:\n"
        "  - {task_id: t1, dataset_id: d901}\n"
        "  - {task_id: t1, dataset_id: d902}\n"
    )
    with pytest.raises(ev.EvaluatorError, match="duplicate"):
        ev.load_tasks_registry(path)


def test_scoreable_tasks_skips_orphans():
    tasks = [
        _task("t1", "d901", "classification", "auroc", "G"),
        _task("t9", "d999", "regression", "pearson", "D"),
    ]
    kept = ev.scoreable_tasks(tasks, {"d901"})
    assert [t["task_id"] for t in kept] == ["t1"]


# --- score_all: offline via injected datasets + tasks + fetch_labels ---


def _two_task_submission(tmp_path, signal=True):
    _, cl, cids, cemb = _classification(signal=signal)
    _, rl, rids, remb = _regression(signal=signal)
    _write_csv(tmp_path / "sub.csv", {"d901": (cids, cemb), "d902": (rids, remb)})
    tasks = [
        _task("t1", "d901", "classification", "auroc", "Gastro"),
        _task("t2", "d902", "regression", "pearson", "Derm"),
    ]
    labels_by = {"t1": cl, "t2": rl}
    fetch = lambda tid, token=None: labels_by[tid]
    datasets = [{"id": "d901"}, {"id": "d902"}]
    return tmp_path / "sub.csv", datasets, tasks, fetch


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


def test_score_all_scopes_submission_to_selected_modality(tmp_path):
    task, labels, ids, emb = _classification()
    _write_csv(tmp_path / "sub.csv", {"d901": (ids, emb)})
    tasks = [
        task,
        _task("t925", "d917", "regression", "pearson", "Infectious diseases"),
    ]
    datasets = [
        {"id": "d901", "modality": "bulk RNA"},
        {"id": "d917", "modality": "Single Cell RNA"},
    ]
    result = ev.score_all(
        tmp_path / "sub.csv",
        modality="bulk-rna",
        datasets=datasets,
        tasks=tasks,
        fetch_labels=lambda task_id, token=None: labels,
    )
    assert result["n_datasets_total"] == 1
    assert result["full_coverage"] is True
    with pytest.raises(ev.SubmissionError, match="outside the selected modality"):
        ev.score_all(
            tmp_path / "sub.csv",
            modality="single-cell-rna",
            datasets=datasets,
            tasks=tasks,
            fetch_labels=_no_fetch,
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
        _task("t1", "d901", "classification", "auroc", "Gastro"),
        _task("t2", "d901", "regression", "pearson", "Gastro"),
    ]
    labels_by = {"t1": cl, "t2": rl}
    fetch = lambda tid, token=None: labels_by[tid]
    res = ev.score_all(
        path=tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=fetch,
    )
    assert res["n_total"] == 2 and res["n_scored"] == 2
    assert res["full_coverage"] is True
    assert res["n_datasets_total"] == 1 and res["n_datasets_scored"] == 1
    assert set(res["categories"]) == {"treatment_outcome", "clinical_scores"}
    assert res["incomplete"] == []


def test_compatible_regression_tasks_share_spectral_factorizations(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(82)
    n = 60
    ids = [f"s{i}" for i in range(n)]
    matrix = rng.normal(size=(n, 8))
    folds = _folds(n)
    labels = [
        _labels(matrix[:, 0] + rng.normal(scale=0.1, size=n), folds, ids),
        _labels(matrix[:, 1] + rng.normal(scale=0.1, size=n), folds, ids),
    ]
    tasks = [
        _task("t1", "d901", "regression", "pearson", "Gastro"),
        _task("t2", "d901", "regression", "pearson", "Gastro"),
    ]
    frame = ev._l2_normalize_rows(_emb_frame(matrix, ids))
    expected = [
        ev.score_task(task, target, frame) for task, target in zip(tasks, labels)
    ]
    _write_csv(tmp_path / "s.csv", {"d901": (ids, matrix)})
    calls = 0
    original = ev._ridge_fold_target_losses

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ev, "_ridge_fold_target_losses", counted)
    result = ev.score_all(
        tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=lambda task_id, token=None: labels[int(task_id[-1]) - 1],
    )

    assert [score.score for score in result["per_task"]] == pytest.approx(
        [score.score for score in expected]
    )
    assert calls == 25


def test_missing_dataset_makes_all_its_tasks_missing(tmp_path):
    _, cl, cids, cemb = _classification()
    _write_csv(tmp_path / "s.csv", {"d901": (cids, cemb)})
    tasks = [
        _task("t1", "d901", "classification", "auroc", "Gastro"),
        _task("t2", "d902", "regression", "pearson", "Derm"),
    ]
    fetch = lambda tid, token=None: {"t1": cl}[tid]
    res = ev.score_all(
        path=tmp_path / "s.csv",
        datasets=[{"id": "d901"}, {"id": "d902"}],
        tasks=tasks,
        fetch_labels=fetch,
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
        _task("t1", "d901", "classification", "auroc", "Gastro"),
        _task("t2", "d901", "classification", "auroc", "Gastro"),
    ]
    bad = _labels(
        np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3]), ids=["z0", "z1", "z2", "z3"]
    )
    fetch = lambda tid, token=None: {"t1": cl, "t2": bad}[tid]
    res = ev.score_all(
        path=tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=fetch,
    )
    assert res["n_scored"] == 1 and res["full_coverage"] is False
    invalid = res["invalid"][0]
    assert invalid["task_id"] == "t2" and invalid["dataset_id"] == "d901"
    assert res["incomplete"] == ["d901"]
    assert res["n_datasets_scored"] == 0


def test_block_invalid_fails_all_its_tasks(tmp_path):
    df = pd.DataFrame(
        {
            ev.DATASET_ID: ["d901", "d901"],
            ev.SAMPLE_ID: ["a", "a"],
            "e0": [1.0, 2.0],
            "e1": [3.0, 4.0],
        }
    )
    df.to_csv(tmp_path / "s.csv", index=False)
    tasks = [
        _task("t1", "d901", "classification", "auroc", "Gastro"),
        _task("t2", "d901", "regression", "pearson", "Gastro"),
    ]
    res = ev.score_all(
        path=tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=_no_fetch,
    )
    assert res["n_scored"] == 0
    reasons = {i["task_id"]: i["reason"] for i in res["invalid"]}
    assert set(reasons) == {"t1", "t2"}
    assert "duplicate" in reasons["t1"] and reasons["t1"] == reasons["t2"]
    assert res["incomplete"] == ["d901"]


def test_score_all_unknown_dataset_raises(tmp_path):
    _, _cl, cids, cemb = _classification()
    _write_csv(
        tmp_path / "s.csv", {"d901": (cids, cemb), "d999": (["a"], np.ones((1, 8)))}
    )
    tasks = [_task("t1", "d901", "classification", "auroc", "Gastro")]
    with pytest.raises(ev.SubmissionError, match="unknown"):
        ev.score_all(
            path=tmp_path / "s.csv",
            datasets=[{"id": "d901"}],
            tasks=tasks,
            fetch_labels=_no_fetch,
        )


def test_incomplete_dataset_is_invalid(tmp_path):
    _, cl, cids, cemb = _classification(n=60)
    _write_csv(tmp_path / "s.csv", {"d901": (cids[:-5], cemb[:-5])})
    tasks = [_task("t1", "d901", "classification", "auroc", "Gastro")]
    fetch = lambda tid, token=None: cl
    res = ev.score_all(
        path=tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=fetch,
    )
    assert res["n_scored"] == 0
    assert res["invalid"][0]["task_id"] == "t1"
    assert "missing" in res["invalid"][0]["reason"]
    assert res["incomplete"] == ["d901"]


def test_scoring_error_is_invalid(tmp_path):
    n = 60
    folds = np.repeat(np.arange(5), 12)
    y = np.zeros(n, dtype=int)
    y[:2] = 1
    ids = [f"s{i}" for i in range(n)]
    emb = np.random.default_rng(0).normal(size=(n, 6))
    labels = _labels(y, folds, ids)
    _write_csv(tmp_path / "s.csv", {"d901": (ids, emb)})
    tasks = [_task("t1", "d901", "classification", "auroc", "Gastro")]
    fetch = lambda tid, token=None: labels
    res = ev.score_all(
        path=tmp_path / "s.csv",
        datasets=[{"id": "d901"}],
        tasks=tasks,
        fetch_labels=fetch,
    )
    assert res["n_scored"] == 0
    assert "single class" in res["invalid"][0]["reason"]


def test_fetch_failure_raises_evaluator_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "FETCH_BACKOFF", 0.0)
    _, _cl, cids, cemb = _classification()
    _write_csv(tmp_path / "s.csv", {"d901": (cids, cemb)})
    tasks = [_task("t1", "d901", "classification", "auroc", "Gastro")]

    def boom(task_id, token=None):
        raise ConnectionError("HF unreachable")

    with pytest.raises(ev.EvaluatorError, match="could not load"):
        ev.score_all(
            path=tmp_path / "s.csv",
            datasets=[{"id": "d901"}],
            tasks=tasks,
            fetch_labels=boom,
        )


def test_unknown_metric_raises_evaluator_error(tmp_path):
    _, cl, cids, cemb = _classification()
    _write_csv(tmp_path / "s.csv", {"d901": (cids, cemb)})
    tasks = [_task("t1", "d901", "classification", "bogus", "Gastro")]
    fetch = lambda tid, token=None: cl
    with pytest.raises(ev.EvaluatorError, match="unknown metric"):
        ev.score_all(
            path=tmp_path / "s.csv",
            datasets=[{"id": "d901"}],
            tasks=tasks,
            fetch_labels=fetch,
        )


# --- transfer: fit once on the train cohort, score the test cohort ---


def _transfer_labels(y, split, ids=None):
    ids = ids if ids is not None else [f"s{i}" for i in range(len(y))]
    return pd.DataFrame(
        {
            ev.SAMPLE_ID: ids,
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
    task = _task("t916", "d910", "classification", "auroc", "Gastro")
    return task, _transfer_labels(y, split, ids), ids, emb


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
    task = _task("t916", "d910", "classification", "auroc", "Gastro")
    with pytest.raises(ev.SubmissionError):
        ev.score_task(task, _transfer_labels(y, split, ids), _emb_frame(emb, ids))


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
            ev.SAMPLE_ID: ["a"],
            ev.SUBJECT_ID: ["p1"],
            ev.LABEL: [0],
            ev.FOLD: [0],
            ev.SPLIT: ["train"],
        }
    )
    both.to_csv(tmp_path / "both.csv", index=False)
    with pytest.raises(ev.EvaluatorError, match="exactly one"):
        ev.load_labels(tmp_path / "both.csv")
    neither = pd.DataFrame({ev.SAMPLE_ID: ["a"], ev.SUBJECT_ID: ["p1"], ev.LABEL: [0]})
    neither.to_csv(tmp_path / "neither.csv", index=False)
    with pytest.raises(ev.EvaluatorError, match="exactly one"):
        ev.load_labels(tmp_path / "neither.csv")


def test_score_all_transfer_two_tasks(tmp_path):
    _, l16, ids, emb = _transfer_classification(signal=True)
    y = l16[ev.LABEL].to_numpy()
    rev = np.where(
        l16[ev.SPLIT].to_numpy() == ev.SPLIT_TRAIN, ev.SPLIT_TEST, ev.SPLIT_TRAIN
    )
    l17 = _transfer_labels(y, rev, ids)
    _write_csv(tmp_path / "sub.csv", {"d910": (ids, emb)})
    tasks = [
        _task("t916", "d910", "classification", "auroc", "Gastro"),
        _task("t917", "d910", "classification", "auroc", "Gastro"),
    ]
    labels_by = {"t916": l16, "t917": l17}
    fetch = lambda tid, token=None: labels_by[tid]
    res = ev.score_all(
        path=tmp_path / "sub.csv",
        datasets=[{"id": "d910"}],
        tasks=tasks,
        fetch_labels=fetch,
    )
    assert res["n_total"] == 2 and res["n_scored"] == 2
    assert res["full_coverage"] is True
    assert res["n_datasets_total"] == 1 and res["n_datasets_scored"] == 1
    assert set(res["categories"]) == {"treatment_outcome"}


# --- perturbation response decoding ---


def _perturbation_task(
    task_id="t918", dataset_id="d911", pairing="paired", target_centered=False
):
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "task_type": ev.PERTURBATION,
        "metric": ev.CENTERED_SPEARMAN,
        "category": "perturbation_prediction",
        "therapeutic_area": "Dermatology",
        "pairing": pairing,
        "n_top_de_genes": 4,
        "target_centered": target_centered,
    }


def _perturbation_labels(ids, folds):
    return pd.DataFrame(
        {
            ev.SAMPLE_ID: ids,
            ev.SUBJECT_ID: [f"p{i}" for i in range(len(ids))],
            ev.PAIR_ID: [f"p{i}" for i in range(len(ids))],
            ev.PAIRING: ev.PAIRED,
            ev.FOLD: folds,
            f"{ev.FOLD}_1": np.roll(folds, 1),
            f"{ev.FOLD}_2": np.roll(folds, 2),
        }
    )


def test_perturbation_constant_embedding_anchors_at_zero():
    rng = np.random.default_rng(12)
    n = 30
    ids = [f"s{i}" for i in range(n)]
    labels = _perturbation_labels(ids, _folds(n, 3))
    targets = ev.PerturbationTargets(
        delta=rng.normal(size=(n, 8)), gene_ids=np.array([f"g{i}" for i in range(8)])
    )
    result = ev.score_task(
        _perturbation_task(), labels, _emb_frame(np.ones((n, 3)), ids), targets
    )
    assert result.score == pytest.approx(0.0, abs=1e-12)
    assert len(result.repeat_scores) == 3
    assert result.diagnostics == {}


def test_perturbation_signal_decodes_hidden_response():
    rng = np.random.default_rng(4)
    n = 45
    ids = [f"s{i}" for i in range(n)]
    emb = rng.normal(size=(n, 5))
    weights = rng.normal(size=(5, 8))
    delta = emb @ weights + rng.normal(scale=0.01, size=(n, 8))
    labels = _perturbation_labels(ids, _folds(n, 3))
    targets = ev.PerturbationTargets(delta, np.array([f"g{i}" for i in range(8)]))
    result = ev.score_task(_perturbation_task(), labels, _emb_frame(emb, ids), targets)
    assert result.score > 0.9
    assert result.diagnostics == {}


def test_perturbation_decoder_clips_test_coordinates(monkeypatch):
    seen = {}

    def ridge_predict(x_train, y_train, x_test, groups_train):
        seen["maximum"] = float(np.max(np.abs(x_test)))
        return np.tile(y_train.mean(axis=0), (len(x_test), 1))

    monkeypatch.setattr(ev, "_ridge_predict", ridge_predict)
    matrix = np.array([[0.0], [1.0], [1_000_000.0]])
    delta = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    score = ev._perturbation_fold(
        matrix,
        delta,
        np.array(["p1", "p2", "p3"]),
        np.array([True, True, False]),
        np.array([False, False, True]),
        2,
        False,
    )

    assert seen["maximum"] == ev.PERTURBATION_TEST_Z_CLIP
    assert score == pytest.approx(0.0)


def test_deg_selection_uses_only_the_supplied_training_rows():
    delta = np.array(
        [
            [8.0, 0.0],
            [8.0, 0.0],
            [0.0, 100.0],
            [0.0, 100.0],
        ]
    )
    source = np.array([True, True, False, False])
    assert ev._select_degs(delta[source], 1).tolist() == [0]
    assert ev._select_degs(delta[~source], 1).tolist() == [1]


def test_directional_transfer_never_fits_on_target_disease(monkeypatch):
    matrix = np.arange(24, dtype=float).reshape(6, 4)
    delta = np.arange(18, dtype=float).reshape(6, 3)
    labels = pd.DataFrame(
        {
            ev.SUBJECT_ID: [f"p{i}" for i in range(6)],
            ev.SPLIT: [ev.SPLIT_TRAIN] * 4 + [ev.SPLIT_TEST] * 2,
        }
    )
    seen = {}

    def fold(matrix_arg, delta_arg, groups_arg, train, test, n_top, target_centered):
        assert groups_arg.tolist() == [f"p{i}" for i in range(6)]
        seen["train"] = np.flatnonzero(train).tolist()
        seen["test"] = np.flatnonzero(test).tolist()
        seen["n_top"] = n_top
        seen["target_centered"] = target_centered
        return 0.75

    monkeypatch.setattr(ev, "_perturbation_fold", fold)
    metrics, n_test = ev._transfer_perturbation_scores(matrix, delta, labels, 2, True)
    assert seen == {
        "train": [0, 1, 2, 3],
        "test": [4, 5],
        "n_top": 2,
        "target_centered": True,
    }
    assert metrics == (0.75,)
    assert n_test == 2


def test_repeated_sample_ids_reuse_one_baseline_embedding():
    labels = pd.DataFrame({ev.SAMPLE_ID: ["s1", "s1", "s2", "s1"]})
    embeddings = _emb_frame(np.array([[1.0, 2.0], [3.0, 4.0]]), ["s1", "s2"])
    aligned = ev._align(labels, embeddings)
    np.testing.assert_array_equal(
        aligned,
        np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
    )


def test_mouse_cartesian_partitions_isolate_both_specimens():
    controls = np.repeat(["c1", "c2", "c3"], 3)
    perturbed = np.tile(["p1", "p2", "p3"], 3)
    repeats = ev._cartesian_partitions(controls, perturbed)
    assert len(repeats) == 6
    assert all(len(folds) == 3 for folds in repeats)
    tested_edges = []
    for folds in repeats:
        for train, test in folds:
            held = np.flatnonzero(test)[0]
            tested_edges.append((controls[held], perturbed[held]))
            assert train.sum() == 4 and test.sum() == 1
            assert not np.any(controls[train] == controls[held])
            assert not np.any(perturbed[train] == perturbed[held])
    assert set(tested_edges) == set(zip(controls, perturbed))
    assert len(tested_edges) == 18


def test_cartesian_perturbation_task_has_six_regular_weighted_repeat_scores():
    rng = np.random.default_rng(7)
    controls = np.repeat(["c1", "c2", "c3"], 3)
    perturbed = np.tile(["p1", "p2", "p3"], 3)
    sample_ids = np.repeat(["s1", "s2", "s3"], 3)
    labels = pd.DataFrame(
        {
            ev.SAMPLE_ID: sample_ids,
            ev.SUBJECT_ID: controls,
            ev.PAIR_ID: [f"pair{i}" for i in range(9)],
            ev.CONTROL_ID: controls,
            ev.PERTURBED_ID: perturbed,
            ev.PAIRING: ev.CARTESIAN,
        }
    )
    embeddings = _emb_frame(np.ones((3, 2)), ["s1", "s2", "s3"])
    targets = ev.PerturbationTargets(
        rng.normal(size=(9, 6)), np.array([f"g{i}" for i in range(6)])
    )
    result = ev.score_task(
        _perturbation_task("t922", "d914", ev.CARTESIAN),
        labels,
        embeddings,
        targets,
    )
    assert result.score == pytest.approx(0.0, abs=1e-12)
    assert len(result.repeat_scores) == 6
    assert result.category == "perturbation_prediction"


def test_load_private_perturbation_artifacts(tmp_path):
    labels = pd.DataFrame(
        {
            ev.SAMPLE_ID: ["s1", "s2"],
            ev.SUBJECT_ID: ["p1", "p2"],
            ev.PAIR_ID: ["p1", "p2"],
            ev.PAIRING: [ev.PAIRED, ev.PAIRED],
            ev.FOLD: [0, 1],
        }
    )
    labels.to_csv(tmp_path / ev.LABELS_FILENAME, index=False)
    np.savez_compressed(
        tmp_path / ev.TARGETS_FILENAME,
        delta=np.ones((2, 3)),
        gene_ids=np.array(["1", "2", "3"]),
    )
    loaded_labels = ev.load_labels(tmp_path / ev.LABELS_FILENAME)
    loaded_targets = ev.load_targets(tmp_path / ev.TARGETS_FILENAME)
    assert ev.LABEL not in loaded_labels
    assert loaded_targets.delta.shape == (2, 3)


def test_score_all_fetches_hidden_targets_only_for_perturbation(tmp_path):
    rng = np.random.default_rng(31)
    n = 30
    ids = [f"s{i}" for i in range(n)]
    emb = rng.normal(size=(n, 4))
    targets = ev.PerturbationTargets(
        emb @ rng.normal(size=(4, 6)), np.array([f"g{i}" for i in range(6)])
    )
    labels = _perturbation_labels(ids, _folds(n, 3))
    _write_csv(tmp_path / "submission.csv", {"d911": (ids, emb)})
    result = ev.score_all(
        tmp_path / "submission.csv",
        datasets=[{"id": "d911"}],
        tasks=[_perturbation_task()],
        fetch_labels=lambda task_id, token=None: labels,
        fetch_targets=lambda task_id, token=None: targets,
    )
    assert result["full_coverage"] is True
    assert (
        result["categories"]["perturbation_prediction"]["metric"]
        == ev.CENTERED_SPEARMAN
    )
    assert result["per_task"][0].diagnostics == {}


# --- representation: embeddings go on the unit sphere, gene space does not ---


def test_embedding_submission_is_read_on_the_unit_sphere(tmp_path):
    """Rescaling one patient's vector must not move an embedding score."""
    path, datasets, tasks, fetch = _two_task_submission(tmp_path, signal=True)
    frame = pd.read_csv(path)
    scaled = frame.copy()
    dims = [c for c in frame.columns if c not in {ev.DATASET_ID, ev.SAMPLE_ID}]
    scaled.loc[::2, dims] = scaled.loc[::2, dims] * 7.0
    scaled.to_csv(tmp_path / "scaled.csv", index=False)

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
    dims = [c for c in frame.columns if c not in {ev.DATASET_ID, ev.SAMPLE_ID}]
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


# --- reference concatenation: the "+ HVG" entry ---


def _reference_for(path, out, width=3, seed=5):
    """Random gene-space features for every sample of ``path``, rows shuffled."""
    frame = pd.read_csv(path)[[ev.DATASET_ID, ev.SAMPLE_ID]].sample(
        frac=1.0, random_state=seed
    )
    rng = np.random.default_rng(seed)
    for column in range(width):
        frame[f"g{column}"] = rng.lognormal(size=len(frame))
    frame.to_csv(out, index=False)
    return out


def test_reference_is_joined_by_sample_then_the_row_is_unit_norm():
    emb = pd.DataFrame([[0.6, 0.8], [1.0, 0.0]], index=["s0", "s1"])
    reference = pd.DataFrame(
        [[0.0, 2.0], [5.0, 0.0], [1.0, 1.0]], index=["s1", "s0", "s9"]
    )

    joined = ev._concat_reference(emb, reference)

    assert list(joined.index) == ["s0", "s1"]
    assert joined.shape[1] == 4
    np.testing.assert_allclose(np.linalg.norm(joined, axis=1), 1.0)
    np.testing.assert_allclose(joined.loc["s1"], np.array([1, 0, 0, 2]) / np.sqrt(5))


def test_reference_scoring_normalizes_the_embedding_block_first(tmp_path):
    """Rescaling a patient's embedding must not change its weight against HVG."""
    path, datasets, tasks, fetch = _two_task_submission(tmp_path, signal=True)
    reference = _reference_for(path, tmp_path / "reference.csv")
    frame = pd.read_csv(path)
    dims = [c for c in frame.columns if c not in {ev.DATASET_ID, ev.SAMPLE_ID}]
    frame.loc[::2, dims] = frame.loc[::2, dims] * 7.0
    frame.to_csv(tmp_path / "scaled.csv", index=False)

    kwargs = {"datasets": datasets, "tasks": tasks, "fetch_labels": fetch}
    plain = ev.score_all(path, reference=reference, **kwargs)
    stretched = ev.score_all(tmp_path / "scaled.csv", reference=reference, **kwargs)
    alone = ev.score_all(path, **kwargs)

    assert plain["full_coverage"] is True
    assert [round(t.score, 10) for t in plain["per_task"]] == [
        round(t.score, 10) for t in stretched["per_task"]
    ]
    assert [t.score for t in plain["per_task"]] != [t.score for t in alone["per_task"]]


def test_reference_missing_a_dataset_is_our_error(tmp_path):
    path, datasets, tasks, fetch = _two_task_submission(tmp_path)
    reference = _reference_for(path, tmp_path / "reference.csv")
    frame = pd.read_csv(reference)
    frame[frame[ev.DATASET_ID] != "d902"].to_csv(reference, index=False)

    with pytest.raises(ev.EvaluatorError, match="d902"):
        ev.score_all(
            path,
            reference=reference,
            datasets=datasets,
            tasks=tasks,
            fetch_labels=fetch,
        )


def test_reference_missing_a_labelled_sample_is_our_error(tmp_path):
    """A gap in our features must not be blamed on the submission."""
    path, datasets, tasks, fetch = _two_task_submission(tmp_path)
    reference = _reference_for(path, tmp_path / "reference.csv")
    pd.read_csv(reference).iloc[1:].to_csv(reference, index=False)

    with pytest.raises(ev.EvaluatorError, match="reference features lack 1/"):
        ev.score_all(
            path,
            reference=reference,
            datasets=datasets,
            tasks=tasks,
            fetch_labels=fetch,
        )


def test_reference_scores_never_reuse_plain_checkpoints(tmp_path):
    path, _, _, _ = _two_task_submission(tmp_path)
    reference = _reference_for(path, tmp_path / "reference.csv")

    plain = ev._TaskCheckpoints(tmp_path, path, "bulk-rna", ev.EMBEDDING)
    concatenated = ev._TaskCheckpoints(
        tmp_path, path, "bulk-rna", ev.EMBEDDING, reference
    )

    assert plain.directory != concatenated.directory


def test_unknown_representation_is_the_submitters_error(tmp_path):
    path, datasets, tasks, fetch = _two_task_submission(tmp_path)
    with pytest.raises(ev.SubmissionError, match="unknown representation"):
        ev.score_all(
            path,
            representation="whatever",
            datasets=datasets,
            tasks=tasks,
            fetch_labels=fetch,
        )


def test_zero_vector_survives_normalisation(tmp_path):
    """An all-zero sample has no direction; it must not become NaN."""
    frame = ev._l2_normalize_rows(
        pd.DataFrame([[0.0, 0.0], [3.0, 4.0]], index=["s0", "s1"])
    )

    assert frame.loc["s0"].tolist() == [0.0, 0.0]
    assert frame.loc["s1"].tolist() == [0.6, 0.8]


def test_regression_inner_selection_keeps_subjects_on_one_side():
    groups = np.repeat([f"p{i}" for i in range(6)], 2)

    for train, test in ev._group_inner_cv(groups):
        assert set(groups[train]).isdisjoint(groups[test])


def test_spectral_ridge_matches_sklearn_grouped_cv():
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(48, 12))
    target = matrix @ rng.normal(size=12) + rng.normal(scale=0.2, size=48)
    groups = np.repeat([f"p{i}" for i in range(24)], 2)
    cv = ev._group_inner_cv(groups)
    expected = RidgeCV(
        alphas=ev.RIDGE_ALPHAS,
        cv=cv,
        scoring="neg_mean_squared_error",
    ).fit(matrix, target)

    assert ev._ridge_alpha(matrix, target, groups) == expected.alpha_


def test_task_checkpoints_resume_without_refitting(tmp_path, monkeypatch):
    path, datasets, tasks, fetch = _two_task_submission(tmp_path, signal=True)
    checkpoint_dir = tmp_path / "checkpoints"
    first = ev.score_all(
        path,
        datasets=datasets,
        tasks=tasks,
        fetch_labels=fetch,
        checkpoint_dir=checkpoint_dir,
    )

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("a completed task was refit")

    monkeypatch.setattr(ev, "score_task", fail_if_refit)
    resumed = ev.score_all(
        path,
        datasets=datasets,
        tasks=tasks,
        fetch_labels=fetch,
        checkpoint_dir=checkpoint_dir,
    )

    assert [score.score for score in resumed["per_task"]] == [
        score.score for score in first["per_task"]
    ]


def test_classification_inner_selection_is_subject_grouped():
    """The penalty selection never splits a patient across inner folds."""
    groups = np.repeat([f"p{i}" for i in range(8)], 2)
    y = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 2)
    cv = ev._inner_cv(y, groups)

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
    ideal_positives = y.sum() / ev.INNER_CV
    ideal_size = len(y) / ev.INNER_CV

    for _, test in ev._inner_cv(y, groups):
        assert abs(y[test].sum() - ideal_positives) <= 2
        assert abs(len(test) - ideal_size) <= 3


def test_load_labels_requires_subject_ids(tmp_path):
    pd.DataFrame(
        {ev.SAMPLE_ID: ["s1", "s2"], ev.LABEL: [0, 1], ev.FOLD: [0, 1]}
    ).to_csv(tmp_path / "labels.csv", index=False)

    assert ev.fetch_public_task_ids() == set()
    with pytest.raises(ev.EvaluatorError, match="subject_id"):
        ev.load_labels(tmp_path / "labels.csv")


def test_public_task_ids_are_read_from_the_public_manifest(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("tasks:\n  - task_id: t910\n    title: PASI\n  - task_id: t917\n")
    assert ev.public_task_ids_from_manifest(path) == {"t910", "t917"}


@pytest.mark.parametrize(
    "error",
    [EntryNotFoundError(ev.PUBLIC_TASKS_MANIFEST), ConnectionError("Hub down")],
)
def test_no_task_is_public_when_the_manifest_is_missing_or_unreachable(
    monkeypatch, error
):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    assert ev.fetch_public_task_ids() == set()
