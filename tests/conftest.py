"""Shared fixtures: a fake Hub and a local data folder, both made up."""

import json
from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
import yaml
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import filter_repo_objects

from primomics import cli, data

REVISION = "0123456789abcdef"
DATASETS = [
    {
        "id": "d901",
        "modality": "bulk RNA",
        "path": "d901/expression.h5ad",
        "n_samples": 4,
    },
    {
        "id": "d902",
        "modality": "bulk RNA",
        "path": "d902/expression.h5ad",
        "n_samples": 3,
    },
    {
        "id": "d903",
        "modality": "Single Cell RNA",
        "path": "d903/expression.h5ad",
        "n_samples": 2,
        "observation_level": "cell",
        "sample_id_column": "sample_id",
    },
]
PUBLIC_TASKS = [
    {
        "task_id": "t901",
        "dataset_id": "d901",
        "task_type": "classification",
        "metric": "auroc",
        "category": "treatment_outcome",
    },
    {
        "task_id": "t903",
        "dataset_id": "d903",
        "task_type": "classification",
        "metric": "auroc",
        "category": "endotype",
    },
    {
        "task_id": "t905",
        "dataset_id": "d901",
        "task_type": "regression",
        "metric": "pearson",
        "category": "clinical_scores",
    },
]
BULK_SAMPLES = {
    "d901": [f"s{i}" for i in range(60)],
    "d902": [f"r{i}" for i in range(10)],
}
CELL_SAMPLES = ["x2", "x2", "x1", "x1", "x1", "x2"]


class FakeHub:
    """Serves ``files`` as the latest commit of the data repo."""

    def __init__(self, files):
        self.files = files
        self.revision = REVISION

    def dataset_info(self, repo_id):
        return SimpleNamespace(sha=self.revision)

    def list_repo_tree(self, repo_id, recursive=False, revision=None, repo_type=None):
        for path, content in self.files.items():
            lfs = None
            if path.endswith(".h5ad"):
                lfs = {"size": len(content), "oid": f"sha-{path}", "pointerSize": 1}
            yield RepoFile(path=path, size=len(content), oid=f"blob-{path}", lfs=lfs)

    def _write(self, local_dir, path):
        target = Path(local_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[path])
        return str(target)

    def hf_hub_download(
        self, repo_id, filename, *, repo_type=None, revision=None, local_dir=None
    ):
        return self._write(local_dir, filename)

    def snapshot_download(
        self, repo_id, *, repo_type=None, revision=None, local_dir=None, **kwargs
    ):
        for path in filter_repo_objects(
            list(self.files), allow_patterns=kwargs.get("allow_patterns")
        ):
            self._write(local_dir, path)
        return str(local_dir)


def hub_files():
    """The made-up repo: three datasets, labels for two of the three public tasks."""
    files = {
        "datasets.yaml": yaml.safe_dump({"datasets": DATASETS}).encode(),
        "README.md": b"card",
        "ATTRIBUTION.md": b"credits",
        "public_tasks/manifest.yaml": b"tasks",
        "public_tasks/t901/labels.csv": b"labels",
        "public_tasks/t903/labels.csv": b"labels",
    }
    for entry in DATASETS:
        files[entry["path"]] = f"matrix {entry['id']}".encode()
        files[f"{entry['id']}/LICENCE.md"] = b"licence"
    return files


@pytest.fixture
def fake_hub(monkeypatch):
    """Route every Hub call of ``primomics.data`` to a ``FakeHub``."""
    hub = FakeHub(hub_files())
    monkeypatch.setattr(data, "HfApi", lambda: hub)
    monkeypatch.setattr(data, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(data, "snapshot_download", hub.snapshot_download)
    return hub


@pytest.fixture
def public_tasks(tmp_path, monkeypatch):
    """Point the CLI at a registry of made-up public tasks."""
    path = tmp_path / "public_tasks.yaml"
    path.write_text(yaml.safe_dump({"tasks": PUBLIC_TASKS}))
    monkeypatch.setattr(cli, "PUBLIC_TASKS", path)
    return PUBLIC_TASKS


def _write_h5ad(path, obs):
    """A tiny AnnData with five genes, one row per entry of ``obs``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.random.default_rng(0).poisson(5.0, size=(len(obs), 5))
    anndata.AnnData(X=counts.astype(np.float32), obs=obs).write_h5ad(path)


def _labels(sample_ids):
    """Binary labels on two repeats of three folds, each fold with both classes."""
    n = len(sample_ids)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "subject_id": [f"p{i}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "fold": [i % 3 for i in range(n)],
            "fold_1": [(i + 1) % 3 for i in range(n)],
        }
    )


@pytest.fixture
def local_data(tmp_path):
    """A data folder as ``primo download`` leaves it, with labels for t901 only."""
    root = tmp_path / "data"
    root.mkdir()
    (root / data.DATASETS_MANIFEST).write_text(yaml.safe_dump({"datasets": DATASETS}))
    for dataset_id, ids in BULK_SAMPLES.items():
        _write_h5ad(root / dataset_id / "expression.h5ad", pd.DataFrame(index=ids))
    cells = pd.DataFrame(
        {"sample_id": pd.Categorical(CELL_SAMPLES)},
        index=[f"c{i}" for i in range(len(CELL_SAMPLES))],
    )
    _write_h5ad(root / "d903" / "expression.h5ad", cells)
    labels = root / data.PUBLIC_TASKS_DIR / "t901" / data.LABELS_FILE
    labels.parent.mkdir(parents=True)
    _labels(BULK_SAMPLES["d901"]).to_csv(labels, index=False)
    lock = {
        path: {"revision": REVISION, "hash": f"hash-{path}"}
        for path in (
            "datasets.yaml",
            "d901/expression.h5ad",
            "d902/expression.h5ad",
            "public_tasks/t901/labels.csv",
        )
    }
    (root / data.LOCK_FILE).write_text(
        json.dumps({"repo": data.DATA_REPO, "files": lock})
    )
    return root


def write_submission(path, blocks):
    """Write ``{dataset_id: (sample_ids, matrix)}`` as one submission file."""
    frames = []
    for dataset_id, (ids, matrix) in blocks.items():
        frame = pd.DataFrame(matrix, columns=[f"e{j}" for j in range(matrix.shape[1])])
        frame.insert(0, "sample_id", ids)
        frame.insert(0, "dataset_id", dataset_id)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(path)
    return path


def signal_blocks(seed=0):
    """Embeddings of d901 that carry its binary label, and noise for d902."""
    rng = np.random.default_rng(seed)
    ids = BULK_SAMPLES["d901"]
    matrix = rng.normal(size=(len(ids), 4))
    matrix[:, 0] += 3.0 * np.array([i % 2 for i in range(len(ids))])
    noise = rng.normal(size=(len(BULK_SAMPLES["d902"]), 4))
    return {"d901": (ids, matrix), "d902": (BULK_SAMPLES["d902"], noise)}
