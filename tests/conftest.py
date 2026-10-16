"""Shared fixtures: a fake Hub that serves a made-up benchmark."""

from pathlib import Path
from types import SimpleNamespace

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
    {"task_id": "t901", "dataset_id": "d901"},
    {"task_id": "t903", "dataset_id": "d903"},
    {"task_id": "t905", "dataset_id": "d901"},
]


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
