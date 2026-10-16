import json
import logging

import pytest
from conftest import DATASETS, PUBLIC_TASKS, REVISION

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


def test_download_counts_the_bytes_it_fetched(fake_hub, tmp_path):
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


def test_a_failed_hub_call_is_an_evaluator_error(fake_hub, tmp_path, monkeypatch):
    def offline(repo_id):
        raise ConnectionError("no network")

    monkeypatch.setattr(fake_hub, "dataset_info", offline)

    with pytest.raises(EvaluatorError, match="no network"):
        data.download(tmp_path, "bulk-rna", tasks=PUBLIC_TASKS)


def test_load_manifest_without_datasets_raises(tmp_path):
    path = tmp_path / data.DATASETS_MANIFEST
    path.write_text("datasets: []\n")

    with pytest.raises(EvaluatorError, match="no datasets"):
        data.load_manifest(path)
