"""The public benchmark data: download it from the Hub and read it locally.

Every dataset is one ``expression.h5ad`` listed in ``datasets.yaml``. The public
tasks also ship their labels under ``public_tasks/``. A download always takes the
latest commit, and records a hash per file in ``primo-lock.json``, so a score can
always be traced back to the exact files it was computed on.
"""

import json
import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import filter_repo_objects

from primomics.errors import EvaluatorError, SubmissionError
from primomics.evaluate import MODALITIES, TASK_ID
from primomics.submission import DATASET_ID, norm_id

DATA_REPO = "PRIMOmics/primo"
REPO_TYPE = "dataset"
DATASETS_MANIFEST = "datasets.yaml"
CARD_FILES = ("README.md", "ATTRIBUTION.md")
PUBLIC_TASKS_DIR = "public_tasks"
LABELS_FILE = "labels.csv"
LOCK_FILE = "primo-lock.json"
DEFAULT_DATA_DIR = "primo-data"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Download:
    """What one download fetched: the commit, the datasets and their public labels."""

    revision: str
    datasets: list[dict]
    labelled: dict[str, list[str]]
    unlabelled: list[str]
    n_bytes: int


@contextmanager
def _hub_errors() -> Iterator[None]:
    """Report a failed Hub call as ours, never as the user's."""
    try:
        yield
    except (SubmissionError, EvaluatorError):
        raise
    except Exception as error:
        raise EvaluatorError(
            f"could not read {DATA_REPO} on the Hub: {error}"
        ) from error


def load_manifest(path: str | Path) -> list[dict]:
    """Read ``datasets.yaml``: one entry per dataset, with its id, modality and path."""
    with open(path) as handle:
        data = yaml.safe_load(handle)
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not datasets:
        raise EvaluatorError(f"no datasets in the manifest: {path}")
    return datasets


def select_datasets(
    manifest: list[dict],
    modality: str,
    tasks: list[dict],
    public_only: bool = False,
    dataset_ids: list[str] | None = None,
) -> list[dict]:
    """The manifest entries of one modality, optionally narrowed, in manifest order."""
    if modality not in MODALITIES:
        raise SubmissionError(
            f"unknown modality '{modality}'; expected one of {sorted(MODALITIES)}"
        )
    in_modality = [
        entry for entry in manifest if entry.get("modality") == MODALITIES[modality]
    ]
    if public_only:
        public = {norm_id(task[DATASET_ID]) for task in tasks}
        in_modality = [entry for entry in in_modality if norm_id(entry["id"]) in public]
    if dataset_ids is None:
        return in_modality
    wanted = {norm_id(dataset_id) for dataset_id in dataset_ids}
    known = {norm_id(entry["id"]) for entry in manifest}
    unknown = sorted(wanted - known)
    if unknown:
        raise SubmissionError(f"unknown dataset id(s): {unknown}")
    selected = [entry for entry in in_modality if norm_id(entry["id"]) in wanted]
    outside = sorted(wanted - {norm_id(entry["id"]) for entry in selected})
    if outside:
        raise SubmissionError(
            f"dataset id(s) not in the {modality} selection: {outside}"
        )
    return selected


def _folder(entry: dict) -> str:
    """The Hub folder that holds one dataset's files."""
    return str(PurePosixPath(entry["path"]).parent)


def _file_hash(file: RepoFile) -> str:
    """The LFS sha256 of a large file, or the git blob id of a small one."""
    return file.lfs.sha256 if file.lfs is not None else file.blob_id


def read_lock(data_dir: str | Path) -> dict:
    """The lock file of a data folder, or an empty lock before any download."""
    path = Path(data_dir) / LOCK_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _update_lock(data_dir: Path, revision: str, files: Iterable[RepoFile]) -> None:
    """Record the commit and the hash of every downloaded file, merged by path."""
    lock = read_lock(data_dir)
    lock["repo"] = DATA_REPO
    lock.setdefault("files", {}).update(
        {file.path: {"revision": revision, "hash": _file_hash(file)} for file in files}
    )
    (data_dir / LOCK_FILE).write_text(json.dumps(lock, indent=2, sort_keys=True))


def _repo_files(api: HfApi, revision: str, patterns: list[str]) -> list[RepoFile]:
    """The files of the data repo at ``revision`` that match ``patterns``."""
    tree = api.list_repo_tree(
        DATA_REPO, recursive=True, revision=revision, repo_type=REPO_TYPE
    )
    return list(
        filter_repo_objects(
            (item for item in tree if isinstance(item, RepoFile)),
            allow_patterns=patterns,
            key=lambda item: item.path,
        )
    )


def _labelled_tasks(
    datasets: list[dict], tasks: list[dict], paths: set[str]
) -> dict[str, list[str]]:
    """The public tasks of each dataset whose labels are among ``paths``."""
    return {
        norm_id(entry["id"]): [
            norm_id(task[TASK_ID])
            for task in tasks
            if norm_id(task[DATASET_ID]) == norm_id(entry["id"])
            and f"{PUBLIC_TASKS_DIR}/{norm_id(task[TASK_ID])}/{LABELS_FILE}" in paths
        ]
        for entry in datasets
    }


def download(
    data_dir: str | Path,
    modality: str,
    *,
    tasks: list[dict],
    public_only: bool = False,
    dataset_ids: list[str] | None = None,
) -> Download:
    """Fetch the latest datasets of one modality, and the labels of its public tasks.

    ``tasks`` are the public tasks; a public task whose labels are not on the Hub
    is reported in ``unlabelled`` rather than failing the download.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    with _hub_errors():
        revision = api.dataset_info(DATA_REPO).sha
        manifest_path = hf_hub_download(
            DATA_REPO,
            DATASETS_MANIFEST,
            repo_type=REPO_TYPE,
            revision=revision,
            local_dir=data_dir,
        )
    datasets = select_datasets(
        load_manifest(manifest_path), modality, tasks, public_only, dataset_ids
    )
    selected = {norm_id(entry["id"]) for entry in datasets}
    task_ids = [
        norm_id(task[TASK_ID])
        for task in tasks
        if norm_id(task[DATASET_ID]) in selected
    ]
    patterns = [
        DATASETS_MANIFEST,
        *CARD_FILES,
        *(f"{_folder(entry)}/*" for entry in datasets),
        *(f"{PUBLIC_TASKS_DIR}/{task_id}/*" for task_id in task_ids),
    ]
    with _hub_errors():
        files = _repo_files(api, revision, patterns)
        snapshot_download(
            DATA_REPO,
            repo_type=REPO_TYPE,
            revision=revision,
            local_dir=data_dir,
            allow_patterns=patterns,
        )
    _update_lock(data_dir, revision, files)
    labelled = _labelled_tasks(datasets, tasks, {file.path for file in files})
    found = {task_id for ids in labelled.values() for task_id in ids}
    unlabelled = [task_id for task_id in task_ids if task_id not in found]
    if unlabelled:
        logger.warning("no public labels on the Hub for: %s", unlabelled)
    return Download(
        revision=revision,
        datasets=datasets,
        labelled=labelled,
        unlabelled=unlabelled,
        n_bytes=sum(file.size for file in files),
    )
