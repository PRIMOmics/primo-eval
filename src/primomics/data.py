"""The public benchmark data: download it from the Hub and read it locally.

Every dataset is one ``expression.h5ad`` listed in ``datasets.yaml``. The public
tasks also ship their labels under ``public_tasks/``. A download always takes the
latest commit, and records a hash per file in ``primo-lock.json``, so a score can
always be traced back to the exact files it was computed on. A submission file is
checked against the local copy before anything is scored or sent.
"""

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import anndata
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import filter_repo_objects

from primomics.errors import EvaluatorError, SubmissionError
from primomics.evaluate import (
    MODALITIES,
    PERTURBATION,
    TASK_ID,
    PerturbationTargets,
    load_labels,
    load_targets,
)
from primomics.submission import (
    DATASET_ID,
    SAMPLE_ID,
    load_submission,
    norm_id,
    to_embedding_frame,
)

DATA_REPO = "PRIMOmics/primo"
REPO_TYPE = "dataset"
DATASETS_MANIFEST = "datasets.yaml"
CARD_FILES = ("README.md", "ATTRIBUTION.md")
PUBLIC_TASKS_DIR = "public_tasks"
LABELS_FILE = "labels.csv"
TARGETS_FILE = "targets.npz"
LOCK_FILE = "primo-lock.json"
DEFAULT_DATA_DIR = "primo-data"
OK = "ok"

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


def local_manifest(data_dir: str | Path) -> list[dict]:
    """The manifest of a data folder that ``primo download`` filled."""
    path = Path(data_dir) / DATASETS_MANIFEST
    if not path.exists():
        raise SubmissionError(f"no data in {data_dir}: run primo download first")
    return load_manifest(path)


def _modality_label(modality: str) -> str:
    """The manifest name of a CLI modality, such as ``bulk RNA`` for ``bulk-rna``."""
    if modality not in MODALITIES:
        raise SubmissionError(
            f"unknown modality '{modality}'; expected one of {sorted(MODALITIES)}"
        )
    return MODALITIES[modality]


def select_datasets(
    manifest: list[dict],
    modality: str,
    tasks: list[dict],
    public_only: bool = False,
    dataset_ids: list[str] | None = None,
) -> list[dict]:
    """The manifest entries of one modality, optionally narrowed, in manifest order."""
    label = _modality_label(modality)
    in_modality = [entry for entry in manifest if entry.get("modality") == label]
    if not in_modality:
        raise EvaluatorError(f"no {modality} dataset in the manifest")
    if public_only:
        public = {norm_id(task[DATASET_ID]) for task in tasks}
        in_modality = [entry for entry in in_modality if norm_id(entry["id"]) in public]
        if not in_modality:
            raise SubmissionError(f"no {modality} dataset carries a public task")
    if dataset_ids is None:
        return in_modality
    wanted = {norm_id(dataset_id) for dataset_id in dataset_ids}
    if not wanted:
        raise SubmissionError("no dataset id given")
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


def _repo_files(
    api: HfApi, revision: str | None, patterns: list[str]
) -> list[RepoFile]:
    """The files of the data repo matching ``patterns``, at ``revision`` or latest."""
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


def _check_hub_covers(datasets: list[dict], paths: set[str]) -> None:
    """Raise if the Hub lacks the file of a selected dataset: that defect is ours."""
    missing = [entry["path"] for entry in datasets if entry["path"] not in paths]
    if missing:
        raise EvaluatorError(
            f"{DATASETS_MANIFEST} lists files missing on the Hub: {missing}"
        )


def _task_files(task: dict) -> list[str]:
    """The repo files a public task needs: labels, and targets for a perturbation."""
    names = [LABELS_FILE]
    if task.get("task_type") == PERTURBATION:
        names.append(TARGETS_FILE)
    return [f"{PUBLIC_TASKS_DIR}/{norm_id(task[TASK_ID])}/{name}" for name in names]


def _labelled_tasks(
    datasets: list[dict], tasks: list[dict], paths: set[str]
) -> dict[str, list[str]]:
    """The public tasks of each dataset whose label files are all among ``paths``."""
    return {
        norm_id(entry["id"]): [
            norm_id(task[TASK_ID])
            for task in tasks
            if norm_id(task[DATASET_ID]) == norm_id(entry["id"])
            and paths.issuperset(_task_files(task))
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

    ``tasks`` are the public tasks; a public task whose label files are not on the
    Hub is reported in ``unlabelled`` rather than failing the download.
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
        paths = {file.path for file in files}
        _check_hub_covers(datasets, paths)
        snapshot_download(
            DATA_REPO,
            repo_type=REPO_TYPE,
            revision=revision,
            local_dir=data_dir,
            allow_patterns=patterns,
        )
    _update_lock(data_dir, revision, files)
    labelled = _labelled_tasks(datasets, tasks, paths)
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


def sample_ids(data_dir: str | Path, entry: dict) -> list[str]:
    """The ids to embed in one dataset: its samples, or its cells' samples."""
    adata = anndata.read_h5ad(Path(data_dir) / entry["path"], backed="r")
    try:
        if entry.get("observation_level") == "cell":
            column = adata.obs[entry.get("sample_id_column", SAMPLE_ID)]
            return [norm_id(value) for value in pd.unique(column)]
        return [norm_id(value) for value in adata.obs_names]
    finally:
        adata.file.close()


def _check_block(
    dataset_id: str,
    block: pd.DataFrame,
    entry: dict | None,
    data_dir: Path,
    modality: str,
) -> str:
    """What is wrong with one dataset of a submission, or ``ok``."""
    if entry is None:
        return "unknown dataset id"
    if entry.get("modality") != _modality_label(modality):
        return f"not a {modality} dataset"
    if not (data_dir / entry["path"]).exists():
        return "not downloaded: run primo download"
    try:
        frame = to_embedding_frame(block)
    except SubmissionError as error:
        return str(error)
    if not np.isfinite(frame.to_numpy()).all():
        return "contains NaN or inf values"
    expected = sample_ids(data_dir, entry)
    present = set(frame.index)
    missing = [sample for sample in expected if sample not in present]
    if missing:
        return f"{len(missing)}/{len(expected)} samples missing, e.g. {missing[:5]}"
    extra = len(present - set(expected))
    if extra:
        logger.warning(
            "%s: %d sample id(s) are not in the dataset and are ignored",
            dataset_id,
            extra,
        )
    return OK


def _read_submission(path: str | Path) -> dict[str, pd.DataFrame]:
    """The blocks of a submission file; a file that cannot be read is the user's."""
    try:
        return load_submission(path)
    except SubmissionError:
        raise
    except (OSError, ValueError) as error:
        raise SubmissionError(f"cannot read {path}: {error}") from error


def check_submission(
    path: str | Path, data_dir: str | Path, modality: str
) -> dict[str, str]:
    """One verdict per dataset of a submission file: ``ok``, or what to fix.

    A dataset passes when every sample of its local ``expression.h5ad`` has one
    finite embedding, which is what scoring needs for every task on it.
    """
    data_dir = Path(data_dir)
    manifest = {norm_id(entry["id"]): entry for entry in local_manifest(data_dir)}
    return {
        dataset_id: _check_block(
            dataset_id, block, manifest.get(dataset_id), data_dir, modality
        )
        for dataset_id, block in _read_submission(path).items()
    }


def _task_dir(data_dir: Path, task_id: str) -> Path:
    """The local folder of one public task."""
    return data_dir / PUBLIC_TASKS_DIR / norm_id(task_id)


def labelled_tasks(data_dir: str | Path, tasks: list[dict]) -> list[dict]:
    """The public tasks whose label files are all in the data folder."""
    data_dir = Path(data_dir)
    return [
        task
        for task in tasks
        if all((data_dir / path).exists() for path in _task_files(task))
    ]


def local_fetchers(
    data_dir: str | Path,
) -> tuple[Callable[[str], pd.DataFrame], Callable[[str], PerturbationTargets]]:
    """Readers of the local label and response files, as ``score_all`` takes them."""
    data_dir = Path(data_dir)

    def fetch_labels(task_id: str) -> pd.DataFrame:
        return load_labels(_task_dir(data_dir, task_id) / LABELS_FILE)

    def fetch_targets(task_id: str) -> PerturbationTargets:
        return load_targets(_task_dir(data_dir, task_id) / TARGETS_FILE)

    return fetch_labels, fetch_targets


def locked_files(
    data_dir: str | Path, datasets: list[dict], task_ids: Iterable[str]
) -> dict[str, dict]:
    """The lock entries of the manifest, of ``datasets`` and of ``task_ids``."""
    files = read_lock(data_dir).get("files", {})
    prefixes = tuple(
        [f"{_folder(entry)}/" for entry in datasets]
        + [f"{PUBLIC_TASKS_DIR}/{norm_id(task_id)}/" for task_id in task_ids]
    )
    return {
        path: entry
        for path, entry in files.items()
        if path == DATASETS_MANIFEST or path.startswith(prefixes)
    }


def changed_datasets(data_dir: str | Path, datasets: list[dict]) -> list[str]:
    """The datasets whose ``expression.h5ad`` is not the one on the Hub today.

    A dataset missing from the lock counts as changed: nothing proves its file is
    the latest one.
    """
    files = read_lock(data_dir).get("files", {})
    paths = [entry["path"] for entry in datasets]
    with _hub_errors():
        latest = {
            file.path: _file_hash(file) for file in _repo_files(HfApi(), None, paths)
        }
    return [
        norm_id(entry["id"])
        for entry in datasets
        if files.get(entry["path"], {}).get("hash") != latest.get(entry["path"])
    ]
