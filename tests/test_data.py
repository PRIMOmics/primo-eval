import json
import logging

import numpy as np
import pytest
from conftest import (
    BULK_SAMPLES,
    DATASETS,
    PUBLIC_TASKS,
    REVISION,
    signal_blocks,
    write_submission,
)

from primomics import data
from primomics.errors import EvaluatorError, SubmissionError


def _downloaded(root):
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


def test_download_fetches_one_modality_and_its_public_labels(fake_hub, tmp_path):
    result = data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS)

    assert _downloaded(tmp_path) == [
        "ATTRIBUTION.md",
        "README.md",
        "d901/LICENCE.md",
        "d901/expression.h5ad",
        "d902/LICENCE.md",
        "d902/expression.h5ad",
        "datasets.yaml",
        data.LOCK_FILE,
        "public_tasks/t901/labels.csv",
    ]
    assert result.revision == REVISION
    assert [entry["id"] for entry in result.datasets] == ["d901", "d902"]
    assert result.labelled == {"d901": ["t901"], "d902": []}
    assert result.unlabelled == ["t905"]


def test_download_reports_the_size_of_the_selection(fake_hub, tmp_path):
    result = data.download(tmp_path, "single-cell-rna", tasks=PUBLIC_TASKS)

    expected = [
        "ATTRIBUTION.md",
        "README.md",
        "d903/LICENCE.md",
        "d903/expression.h5ad",
        "datasets.yaml",
        "public_tasks/t903/labels.csv",
    ]
    assert result.n_bytes == sum(len(fake_hub.files[path]) for path in expected)


def test_public_only_skips_datasets_without_public_tasks(fake_hub, tmp_path):
    result = data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS, public_only=True)

    assert [entry["id"] for entry in result.datasets] == ["d901"]
    assert not (tmp_path / "d902").exists()


def test_dataset_ids_narrow_the_selection(fake_hub, tmp_path):
    result = data.download(
        tmp_path, "bulk-rna", tasks=PUBLIC_TASKS, dataset_ids=["d902"]
    )

    assert [entry["id"] for entry in result.datasets] == ["d902"]
    assert not (tmp_path / "public_tasks").exists()
    assert result.unlabelled == []


def test_unknown_dataset_ids_raise(fake_hub, tmp_path):
    with pytest.raises(SubmissionError, match="unknown dataset id"):
        data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS, dataset_ids=["d999"])


def test_dataset_ids_of_another_modality_raise(fake_hub, tmp_path):
    with pytest.raises(SubmissionError, match="not in the bulk-rna selection"):
        data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS, dataset_ids=["d903"])


def test_unknown_modality_raises():
    with pytest.raises(SubmissionError, match="unknown modality"):
        data.select_datasets(DATASETS, "proteomics", PUBLIC_TASKS)


def test_a_modality_missing_from_the_manifest_raises():
    bulk_only = [entry for entry in DATASETS if entry["modality"] == "bulk RNA"]

    with pytest.raises(EvaluatorError, match="no single-cell-rna dataset"):
        data.select_datasets(bulk_only, "single-cell-rna", PUBLIC_TASKS)


def test_public_only_without_public_datasets_raises():
    with pytest.raises(SubmissionError, match="public task"):
        data.select_datasets(DATASETS, "bulk-rna", [], public_only=True)


def test_an_empty_list_of_dataset_ids_raises():
    with pytest.raises(SubmissionError, match="no dataset id"):
        data.select_datasets(DATASETS, "bulk-rna", PUBLIC_TASKS, dataset_ids=[])


def test_the_lock_records_a_hash_and_the_revision_per_file(fake_hub, tmp_path):
    data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS)

    lock = json.loads((tmp_path / data.LOCK_FILE).read_text())
    assert lock["repo"] == data.DATA_REPO
    assert lock["files"]["d901/expression.h5ad"] == {
        "revision": REVISION,
        "hash": "sha-d901/expression.h5ad",
    }
    assert lock["files"]["public_tasks/t901/labels.csv"] == {
        "revision": REVISION,
        "hash": "blob-public_tasks/t901/labels.csv",
    }


def test_a_second_download_merges_into_the_lock(fake_hub, tmp_path):
    data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS)
    fake_hub.revision = "fedcba9876543210"
    data.download(tmp_path, "single-cell-rna", tasks=PUBLIC_TASKS)

    files = data.read_lock(tmp_path)["files"]
    assert files["d901/expression.h5ad"]["revision"] == REVISION
    assert files["d903/expression.h5ad"]["revision"] == "fedcba9876543210"


def test_read_lock_before_any_download_is_empty(tmp_path):
    assert data.read_lock(tmp_path) == {}


def test_public_tasks_without_labels_are_logged(fake_hub, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS)

    assert "t905" in caplog.text


def test_a_perturbation_task_needs_its_targets(fake_hub, tmp_path):
    tasks = [
        {"task_id": "t901", "dataset_id": "d901", "task_type": "perturbation"},
        {"task_id": "t905", "dataset_id": "d901", "task_type": "perturbation"},
    ]
    fake_hub.files["public_tasks/t901/targets.npz"] = b"targets"
    fake_hub.files["public_tasks/t905/labels.csv"] = b"labels"

    result = data.download(tmp_path, "bulk-rna", tasks=tasks)

    assert result.labelled["d901"] == ["t901"]
    assert result.unlabelled == ["t905"]


def test_a_failed_hub_call_is_an_evaluator_error(fake_hub, tmp_path, monkeypatch):
    def offline(repo_id):
        raise ConnectionError("no network")

    monkeypatch.setattr(fake_hub, "dataset_info", offline)

    with pytest.raises(EvaluatorError, match="no network"):
        data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS)


def test_a_dataset_missing_on_the_hub_raises(fake_hub, tmp_path):
    del fake_hub.files["d902/expression.h5ad"]

    with pytest.raises(EvaluatorError, match="d902/expression.h5ad"):
        data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS)

    assert not (tmp_path / "d901").exists()


def test_load_manifest_without_datasets_raises(tmp_path):
    path = tmp_path / data.DATASETS_MANIFEST
    path.write_text("datasets: []\n")

    with pytest.raises(EvaluatorError, match="no datasets"):
        data.load_manifest(path)


def test_sample_ids_of_a_bulk_dataset_are_its_rows(local_data):
    assert data.sample_ids(local_data, DATASETS[1]) == BULK_SAMPLES["d902"]


def test_sample_ids_of_a_single_cell_dataset_are_its_samples_in_order(local_data):
    assert data.sample_ids(local_data, DATASETS[2]) == ["x2", "x1"]


def test_check_passes_a_complete_file(local_data, tmp_path):
    path = write_submission(tmp_path / "sub.parquet", signal_blocks())

    assert data.check_submission(path, local_data, "bulk-rna") == {
        "d901": data.OK,
        "d902": data.OK,
    }


def test_check_reports_missing_samples(local_data, tmp_path):
    blocks = signal_blocks()
    ids, matrix = blocks["d902"]
    blocks["d902"] = (ids[:7], matrix[:7])
    path = write_submission(tmp_path / "sub.parquet", blocks)

    verdicts = data.check_submission(path, local_data, "bulk-rna")

    assert verdicts["d901"] == data.OK
    assert verdicts["d902"].startswith("3/10 samples missing")


def test_check_reports_non_finite_values(local_data, tmp_path):
    blocks = signal_blocks()
    blocks["d902"][1][0, 0] = np.inf
    path = write_submission(tmp_path / "sub.parquet", blocks)

    verdicts = data.check_submission(path, local_data, "bulk-rna")

    assert verdicts["d902"] == "contains NaN or inf values"


def test_check_reports_duplicate_samples(local_data, tmp_path):
    ids = BULK_SAMPLES["d902"]
    blocks = {"d902": ([*ids, ids[0]], np.ones((len(ids) + 1, 2)))}
    path = write_submission(tmp_path / "sub.parquet", blocks)

    verdicts = data.check_submission(path, local_data, "bulk-rna")

    assert "duplicate sample_ids" in verdicts["d902"]


def test_check_reports_unknown_and_other_modality_datasets(local_data, tmp_path):
    blocks = {"d999": (["a", "b"], np.ones((2, 2))), "d903": (["x1"], np.ones((1, 2)))}
    path = write_submission(tmp_path / "sub.parquet", blocks)

    verdicts = data.check_submission(path, local_data, "bulk-rna")

    assert verdicts == {
        "d999": "unknown dataset id",
        "d903": "not a bulk-rna dataset",
    }


def test_check_reports_a_dataset_that_was_not_downloaded(local_data, tmp_path):
    (local_data / "d902" / "expression.h5ad").unlink()
    path = write_submission(tmp_path / "sub.parquet", signal_blocks())

    verdicts = data.check_submission(path, local_data, "bulk-rna")

    assert verdicts["d902"].startswith("not downloaded")


def test_check_warns_on_samples_that_are_not_in_the_dataset(
    local_data, tmp_path, caplog
):
    blocks = signal_blocks()
    ids, matrix = blocks["d902"]
    blocks["d902"] = ([*ids, "extra"], np.vstack([matrix, np.ones((1, 4))]))
    path = write_submission(tmp_path / "sub.parquet", blocks)

    with caplog.at_level(logging.WARNING):
        verdicts = data.check_submission(path, local_data, "bulk-rna")

    assert verdicts["d902"] == data.OK
    assert "d902: 1 sample id(s)" in caplog.text


def test_check_without_downloaded_data_raises(tmp_path):
    path = write_submission(tmp_path / "sub.parquet", signal_blocks())

    with pytest.raises(SubmissionError, match="run primo download"):
        data.check_submission(path, tmp_path / "empty", "bulk-rna")


def test_labelled_tasks_keep_the_tasks_with_local_label_files(local_data):
    perturbation = {
        "task_id": "t901",
        "dataset_id": "d901",
        "task_type": "perturbation",
    }

    assert data.labelled_tasks(local_data, PUBLIC_TASKS) == [PUBLIC_TASKS[0]]
    assert data.labelled_tasks(local_data, [perturbation]) == []


def test_local_fetchers_read_the_label_files(local_data):
    fetch_labels, _ = data.local_fetchers(local_data)

    labels = fetch_labels("t901")

    assert list(labels["sample_id"]) == BULK_SAMPLES["d901"]
    assert {"subject_id", "label", "fold", "fold_1"} <= set(labels.columns)


def test_locked_files_keep_the_files_of_the_given_datasets_and_tasks(local_data):
    files = data.locked_files(local_data, [DATASETS[0]], ["t901"])

    assert sorted(files) == ["d901/expression.h5ad", "public_tasks/t901/labels.csv"]
