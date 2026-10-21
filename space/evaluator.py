"""Standalone probe for the PRIMO public benchmark.

Loads one embedding submission for one modality (rows keyed by
``dataset_id`` + ``sample_id``), then scores each TASK = (dataset, target): a
dataset is embedded once and reused across all its tasks. Per task it fits a
fixed probe—linear for scalar targets and multi-output ridge for response
matrices—over repeated frozen CV folds or a fixed transfer split. Binary tasks with
fewer than 30 subjects use leave-pair-out over subject pairs instead. Turning those
predictions into scores + leaderboard aggregates is the job of ``scoring.py``;
this module only reads data and runs the probe.

Failures are split by who caused them: a bad submission raises/records a
``SubmissionError`` (the submitter fixes it); anything on our side, such as a
failed Hugging Face fetch or an unexpected bug, raises ``EvaluatorError`` so it
is never silently charged against the submitter.

Reads the public ``datasets.yaml`` manifest (what the submitter embeds) plus the
private ``tasks.yaml`` registry, each task's private ``labels.csv``, and hidden
perturbation ``targets.npz`` matrices where applicable. Each read is at the commit
that ``versions.py`` pins for the current board version, unless a caller passes
another ``revision``. It imports only ``scoring.py``, ``versions.py`` and the
packages of ``requirements.txt``, so it runs unchanged inside a public Hugging Face
Space.

Deps: numpy, pandas, joblib, scikit-learn, pyyaml. ``huggingface_hub`` is used
only by the fetch helpers / CLI, imported lazily.
"""

import argparse
import hashlib
import itertools
import json
import logging
import os
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, cpu_count, delayed, parallel_backend
from scoring import (
    METRICS,
    TaskScore,
    category_means,
    compute_residual_sample_spearman,
    compute_target_centered_sample_spearman,
)
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from versions import CURRENT

SAMPLE_ID = "sample_id"
SUBJECT_ID = "subject_id"
DATASET_ID = "dataset_id"
TASK_ID = "task_id"
LABEL = "label"
FOLD = "fold"
FOLD_PREFIX = f"{FOLD}_"
SPLIT = "split"
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
PAIR_ID = "pair_id"
CONTROL_ID = "control_id"
PERTURBED_ID = "perturbed_id"
PAIRING = "pairing"
PAIRED = "paired"
CARTESIAN = "cartesian"
PERTURBATION = "perturbation"
RESIDUAL_SPEARMAN = "residual_spearman"
CENTERED_SPEARMAN = "centered_spearman"
ORG = "PRIMOmics"
PUBLIC_REPO = f"{ORG}/primo"
LABELS_REPO = f"{ORG}/primo-labels"
RESULTS_REPO = f"{ORG}/primo-results"
MANIFEST_FILENAME = "datasets.yaml"
PUBLIC_TASKS_MANIFEST = "public_tasks/manifest.yaml"
TASKS_FILENAME = "tasks.yaml"
LABELS_FILENAME = "labels.csv"
TARGETS_FILENAME = "targets.npz"
MODALITIES = {"bulk-rna": "bulk RNA", "single-cell-rna": "Single Cell RNA"}

RIDGE_ALPHAS = np.logspace(-6.0, 6.0, 25)
PERTURBATION_TEST_Z_CLIP = 20.0
LOGREG_CS = np.logspace(-6.0, 6.0, 13)
INNER_CV = 5
INNER_CV_JOBS = 4
OUTER_CV_JOBS = min(10, cpu_count())
OUTER_PARALLEL_MIN_SAMPLES = 30
LOGREG_TOL = 1e-3
MAX_ITER = 5000
RANDOM_STATE = 0
MIN_CV_SUBJECTS = 30
CHECKPOINT_PROTOCOL = "grouped-inner-cv-spectral-ridge-v2"

EMBEDDING = "embedding"
GENE_SPACE = "gene_space"
REPRESENTATIONS = (EMBEDDING, GENE_SPACE)

FETCH_ATTEMPTS = 3
FETCH_BACKOFF = 1.0

NPZ_ID_KEYS = ("sample_ids", "sample_id", "ids")
NPZ_EMB_KEYS = ("embeddings", "embedding", "emb", "X")
NPZ_DSID_KEYS = ("dataset_ids", "dataset_id")
NOT_NPZ = "not a .npz archive: save it with np.savez, or use .csv/.parquet"


class SubmissionError(ValueError):
    """A submission the evaluator cannot score (bad format, missing samples...)."""


class EvaluatorError(RuntimeError):
    """A failure on our side (data fetch, unexpected bug), not the submitter's.

    Raised instead of being recorded as a per-dataset skip, so a transient
    Hugging Face hiccup or an evaluator bug never silently costs a submitter
    their coverage.
    """


@dataclass
class TaskOutcome:
    """What happened to one scoreable task: scored / missing / invalid."""

    task_id: str
    dataset_id: str
    status: str
    score: TaskScore | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PerturbationTargets:
    """Hidden response vectors and their aligned gene identifiers."""

    delta: np.ndarray
    gene_ids: np.ndarray


@dataclass(frozen=True)
class _PendingTask:
    """Fetched task inputs awaiting either individual or compatible batch scoring."""

    position: int
    task: dict
    emb: pd.DataFrame
    labels: pd.DataFrame
    targets: PerturbationTargets | None
    fingerprint: str


def _digest_file(digest: "hashlib._Hash", path: str | Path) -> None:
    """Feed a file's bytes into ``digest`` in chunks."""
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


class _TaskCheckpoints:
    """Persist completed task outcomes under one submission-scoped directory."""

    def __init__(
        self,
        directory: str | Path,
        submission: str | Path,
        modality: str | None,
        representation: str,
        reference: str | Path | None = None,
    ) -> None:
        digest = hashlib.sha256()
        _digest_file(digest, submission)
        digest.update(f"{CHECKPOINT_PROTOCOL}\0{modality}\0{representation}".encode())
        if reference is not None:
            digest.update(b"\0reference\0")
            _digest_file(digest, reference)
        digest.update(
            json.dumps(
                {
                    "ridge_alphas": RIDGE_ALPHAS.tolist(),
                    "logreg_cs": LOGREG_CS.tolist(),
                    "inner_cv": INNER_CV,
                    "logreg_tol": LOGREG_TOL,
                    "max_iter": MAX_ITER,
                    "min_cv_subjects": MIN_CV_SUBJECTS,
                    "random_state": RANDOM_STATE,
                },
                sort_keys=True,
            ).encode()
        )
        self.directory = Path(directory) / digest.hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        safe_id = hashlib.sha256(task_id.encode()).hexdigest()[:16]
        return self.directory / f"task-{safe_id}.json"

    def load(self, task_id: str, fingerprint: str) -> TaskOutcome | None:
        path = self._path(task_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            if payload["fingerprint"] != fingerprint:
                return None
            raw_score = payload.get("score")
            score = None
            if raw_score is not None:
                raw_score["repeat_scores"] = tuple(raw_score["repeat_scores"])
                score = TaskScore(**raw_score)
            return TaskOutcome(
                task_id=payload["task_id"],
                dataset_id=payload["dataset_id"],
                status=payload["status"],
                score=score,
                reason=payload.get("reason"),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            logging.warning("ignoring invalid task checkpoint: %s", path)
            return None

    def save(self, outcome: TaskOutcome, fingerprint: str) -> None:
        path = self._path(outcome.task_id)
        score = None
        if outcome.score is not None:
            score = {
                "task_id": outcome.score.task_id,
                "dataset_id": outcome.score.dataset_id,
                "category": outcome.score.category,
                "metric": outcome.score.metric,
                "score": outcome.score.score,
                "n_samples": outcome.score.n_samples,
                "repeat_scores": list(outcome.score.repeat_scores),
                "diagnostics": outcome.score.diagnostics,
            }
        payload = {
            "fingerprint": fingerprint,
            "task_id": outcome.task_id,
            "dataset_id": outcome.dataset_id,
            "status": outcome.status,
            "score": score,
            "reason": outcome.reason,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(path)


def _task_fingerprint(
    task: dict,
    labels: pd.DataFrame,
    targets: PerturbationTargets | None,
) -> str:
    """Fingerprint every private input that can affect a task checkpoint."""
    digest = hashlib.sha256(
        json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(json.dumps(list(labels.columns)).encode())
    digest.update(json.dumps([str(dtype) for dtype in labels.dtypes]).encode())
    digest.update(pd.util.hash_pandas_object(labels, index=True).values.tobytes())
    if targets is not None:
        digest.update(targets.delta.tobytes())
        digest.update(targets.gene_ids.astype(str).tobytes())
    return digest.hexdigest()


def _regression_geometry(labels: pd.DataFrame) -> str:
    """Identify tasks whose design matrices and evaluation splits are identical."""
    columns = [SAMPLE_ID, SUBJECT_ID, *_fold_columns(labels)]
    if SPLIT in labels:
        columns.append(SPLIT)
    frame = labels[columns]
    digest = hashlib.sha256(json.dumps(columns).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _outcome_from_score(task: dict, score: TaskScore) -> TaskOutcome:
    """Map a completed score to the public scored/invalid outcome contract."""
    task_id = _norm_id(task[TASK_ID])
    dataset_id = _norm_id(task[DATASET_ID])
    if not np.isfinite(score.score):
        return TaskOutcome(
            task_id,
            dataset_id,
            "invalid",
            reason="degenerate score (constant predictions)",
        )
    return TaskOutcome(task_id, dataset_id, "scored", score=score)


def _norm_id(value: object) -> str:
    """Normalise an id so int/str/float spellings of the same id join."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_tasks_registry(path: str | Path) -> list[dict]:
    """Read the private ``tasks.yaml`` registry (one entry per task)."""
    with open(path) as handle:
        data = yaml.safe_load(handle)
    tasks = data.get("tasks", []) if isinstance(data, dict) else (data or [])
    ids = [_norm_id(t[TASK_ID]) for t in tasks]
    if len(set(ids)) != len(ids):
        raise EvaluatorError(f"duplicate task_id in the registry: {ids}")
    return tasks


def _fold_columns(labels: pd.DataFrame) -> list[str]:
    """Return the ordered external-CV fold columns in private labels."""
    repeated = [column for column in labels if column.startswith(FOLD_PREFIX)]
    invalid = [
        column for column in repeated if not column.removeprefix(FOLD_PREFIX).isdigit()
    ]
    if invalid:
        raise EvaluatorError(f"labels.csv has invalid repeat-fold columns: {invalid}")
    first = [FOLD] if FOLD in labels.columns else []
    return [
        *first,
        *sorted(repeated, key=lambda column: int(column.removeprefix(FOLD_PREFIX))),
    ]


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read private labels with repeated CV folds or one transfer split."""
    df = pd.read_csv(path)
    is_perturbation = PAIR_ID in df.columns
    required = (
        {SAMPLE_ID, SUBJECT_ID, PAIR_ID}
        if is_perturbation
        else {SAMPLE_ID, SUBJECT_ID, LABEL}
    )
    missing = required - set(df.columns)
    if missing:
        raise EvaluatorError(f"labels.csv missing columns: {sorted(missing)}")
    _subject_groups(df)
    has_folds = bool(_fold_columns(df))
    has_split = SPLIT in df.columns
    pairing_values = (
        set(df[PAIRING].astype(str)) if is_perturbation and PAIRING in df else set()
    )
    is_cartesian = pairing_values == {CARTESIAN}
    if sum((has_folds, has_split, is_cartesian)) != 1:
        raise EvaluatorError(
            "labels.csv needs exactly one evaluation assignment: folds, transfer "
            "split, or Cartesian pairing."
        )
    if is_cartesian:
        missing = {CONTROL_ID, PERTURBED_ID, PAIRING} - set(df.columns)
        if missing:
            raise EvaluatorError(
                f"Cartesian labels.csv missing columns: {sorted(missing)}"
            )
    return df


def load_targets(path: str | Path) -> PerturbationTargets:
    """Read and validate one task's hidden perturbation responses."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"delta", "gene_ids"} - set(data.files)
        if missing:
            raise EvaluatorError(f"targets.npz missing arrays: {sorted(missing)}")
        delta = np.asarray(data["delta"], dtype=float)
        gene_ids = np.asarray(data["gene_ids"]).astype(str)
    if delta.ndim != 2:
        raise EvaluatorError(f"targets delta must be 2D, got shape {delta.shape}")
    if gene_ids.ndim != 1 or len(gene_ids) != delta.shape[1]:
        raise EvaluatorError("targets gene_ids must align with delta columns")
    if not np.isfinite(delta).all():
        raise EvaluatorError("targets delta contains NaN or inf values")
    return PerturbationTargets(delta=delta, gene_ids=gene_ids)


def load_submission(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a multi-dataset submission into ``{dataset_id: raw_block_frame}``."""
    ext = Path(path).suffix.lower()
    if ext == ".npz":
        frame = _npz_to_frame(path)
    elif ext in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    elif ext in (".tsv", ".txt"):
        frame = pd.read_csv(path, sep="\t")
    elif ext == ".csv":
        frame = pd.read_csv(path)
    else:
        raise SubmissionError(
            f"unsupported submission type '{ext}'. Use .csv/.tsv/.parquet/.npz"
        )
    return _split_by_dataset(frame)


def _npz_to_frame(path: str | Path) -> pd.DataFrame:
    try:
        data = np.load(path, allow_pickle=False)
    except (ValueError, EOFError, zipfile.BadZipFile) as err:
        raise SubmissionError(NOT_NPZ) from err
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise SubmissionError(NOT_NPZ)
    ds_key = next((k for k in NPZ_DSID_KEYS if k in data), None)
    id_key = next((k for k in NPZ_ID_KEYS if k in data), None)
    emb_key = next((k for k in NPZ_EMB_KEYS if k in data), None)
    if ds_key is None or id_key is None or emb_key is None:
        raise SubmissionError(
            f".npz needs a dataset-id array {NPZ_DSID_KEYS}, a sample-id array "
            f"{NPZ_ID_KEYS}, and an embedding array {NPZ_EMB_KEYS}; "
            f"found {list(data.keys())}"
        )
    try:
        emb = np.asarray(data[emb_key])
        sample_ids = np.asarray(data[id_key])
        dataset_ids = np.asarray(data[ds_key])
    except (ValueError, zipfile.BadZipFile) as err:
        raise SubmissionError(
            f"cannot read the .npz arrays ({err}). Save ids as text, e.g. "
            "np.asarray(ids, dtype=str), or use .csv/.parquet"
        ) from err
    if emb.ndim != 2:
        raise SubmissionError(f"embeddings must be 2D, got shape {emb.shape}")
    frame = pd.DataFrame(emb)
    frame.insert(0, SAMPLE_ID, sample_ids)
    frame.insert(0, DATASET_ID, dataset_ids)
    return frame


def _split_by_dataset(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group a submission by ``dataset_id`` into per-dataset blocks (raw frames)."""
    lower = {str(c).lower(): c for c in frame.columns}
    if DATASET_ID not in lower:
        raise SubmissionError(
            f"submission needs a '{DATASET_ID}' column; found {list(frame.columns)}"
        )
    ds_col = lower[DATASET_ID]
    blocks: dict[str, pd.DataFrame] = {}
    for raw_ds, group in frame.groupby(ds_col, sort=False):
        dataset_id = _norm_id(raw_ds)
        if dataset_id in blocks:
            raise SubmissionError(
                f"dataset_id '{dataset_id}' appears under multiple spellings"
            )
        blocks[dataset_id] = group.drop(columns=[ds_col]).dropna(axis=1, how="all")
    if not blocks:
        raise SubmissionError("empty submission")
    return blocks


def _to_embedding_frame(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).lower(): c for c in frame.columns}
    if SAMPLE_ID not in lower:
        raise SubmissionError(
            f"submission needs a '{SAMPLE_ID}' column; found {list(frame.columns)}"
        )
    id_col = lower[SAMPLE_ID]
    ids = [_norm_id(v) for v in frame[id_col]]
    emb = frame.drop(columns=[id_col])
    non_numeric = [c for c in emb.columns if not pd.api.types.is_numeric_dtype(emb[c])]
    if non_numeric:
        raise SubmissionError(
            f"embedding columns must be numeric; non-numeric: {non_numeric[:5]}"
        )
    if emb.shape[1] == 0:
        raise SubmissionError("submission has no embedding columns")
    out = emb.astype(float)
    out.index = ids
    if out.index.has_duplicates:
        dups = out.index[out.index.duplicated()].unique().tolist()
        raise SubmissionError(f"duplicate sample_ids in submission: {dups[:5]}")
    return out


def _l2_normalize_rows(emb: pd.DataFrame) -> pd.DataFrame:
    """Put every sample on the unit sphere; an embedding dimension has no unit."""
    norms = np.linalg.norm(emb.to_numpy(dtype=float), axis=1, keepdims=True)
    return pd.DataFrame(
        emb.to_numpy(dtype=float) / np.where(norms == 0.0, 1.0, norms),
        index=emb.index,
        columns=emb.columns,
    )


def _load_reference(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read the gene-space reference concatenated to submissions, by dataset.

    The reference is ours, so any defect in it is an ``EvaluatorError``.
    """
    try:
        return {
            dataset_id: _to_embedding_frame(block)
            for dataset_id, block in load_submission(path).items()
        }
    except (OSError, ValueError) as error:
        message = f"could not load the reference features: {error}"
        raise EvaluatorError(message) from error


def _concat_reference(emb: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Append the reference columns to each sample, then L2-normalize the row.

    Samples absent from either side are dropped, so a labelled sample the
    submission lacks still fails ``_align`` exactly as without a reference. One
    the reference lacks is caught first by ``_check_reference_covers``.
    """
    joined = emb.add_prefix("model_").join(
        reference.add_prefix("reference_"), how="inner"
    )
    return _l2_normalize_rows(joined)


def _check_reference_covers(
    labels: pd.DataFrame, reference: pd.DataFrame, task_id: str
) -> None:
    """Raise if the reference lacks a labelled sample: that defect is ours."""
    lab_ids = [_norm_id(v) for v in labels[SAMPLE_ID]]
    present = set(reference.index)
    missing = [s for s in lab_ids if s not in present]
    if missing:
        raise EvaluatorError(
            f"reference features lack {len(missing)}/{len(lab_ids)} labelled "
            f"samples of {task_id}, e.g. {missing[:5]}"
        )


def _align(labels: pd.DataFrame, emb: pd.DataFrame) -> np.ndarray:
    """Return the embedding matrix in labels order, or raise if a sample is missing."""
    lab_ids = [_norm_id(v) for v in labels[SAMPLE_ID]]
    present = set(emb.index)
    missing = [s for s in lab_ids if s not in present]
    if missing:
        raise SubmissionError(
            f"{len(missing)}/{len(lab_ids)} labelled samples missing from "
            f"submission, e.g. {missing[:5]}"
        )
    matrix = emb.loc[lab_ids].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise SubmissionError("submission contains NaN or inf values")
    return matrix


def _subject_groups(labels: pd.DataFrame) -> np.ndarray:
    """Return normalized private subject ids, rejecting unusable metadata."""
    if SUBJECT_ID not in labels:
        raise EvaluatorError(f"labels.csv missing columns: ['{SUBJECT_ID}']")
    if labels[SUBJECT_ID].isna().any():
        raise EvaluatorError("labels.csv has missing subject_id values")
    groups = np.array([_norm_id(value) for value in labels[SUBJECT_ID]])
    if np.any(groups == ""):
        raise EvaluatorError("labels.csv has empty subject_id values")
    return groups


def _group_inner_cv(groups_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject-grouped inner folds for ridge penalty selection."""
    n_groups = len(np.unique(groups_train))
    n_splits = min(INNER_CV, n_groups)
    if n_splits < 2:
        raise SubmissionError("an outer training fold has fewer than two subjects")
    splitter = GroupKFold(n_splits=n_splits)
    rows = np.zeros(len(groups_train))
    return list(splitter.split(rows, groups=groups_train))


def _ridge_fold_losses(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Return validation MSE for every alpha from one exact SVD."""
    return np.mean(_ridge_fold_target_losses(x_train, y_train, x_test, y_test), axis=1)


def _ridge_fold_target_losses(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Return per-target validation MSE while sharing one factorization."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_test = y_test[:, None]
    x_offset = np.mean(x_train, axis=0)
    y_offset = np.mean(y_train, axis=0)
    x_centered = x_train - x_offset
    y_centered = y_train - y_offset
    u, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    projected_x = (x_test - x_offset) @ vt.T
    projected_y = u.T @ y_centered
    shrinkage = singular_values[:, None] / (
        singular_values[:, None] ** 2 + RIDGE_ALPHAS[None, :]
    )
    losses = []
    for index in range(len(RIDGE_ALPHAS)):
        predictions = projected_x @ (shrinkage[:, index, None] * projected_y) + y_offset
        losses.append(np.mean((predictions - y_test) ** 2, axis=0))
    return np.asarray(losses)


def _ridge_alpha(matrix: np.ndarray, target: np.ndarray, groups: np.ndarray) -> float:
    """Select the grouped-CV ridge penalty without refitting for every alpha."""
    losses = [
        _ridge_fold_losses(matrix[train], target[train], matrix[test], target[test])
        for train, test in _group_inner_cv(groups)
    ]
    return float(RIDGE_ALPHAS[np.argmin(np.mean(losses, axis=0))])


def _ridge_alphas_by_target(
    matrix: np.ndarray, targets: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Select independent penalties while sharing each fold's factorization."""
    losses = [
        _ridge_fold_target_losses(
            matrix[train], targets[train], matrix[test], targets[test]
        )
        for train, test in _group_inner_cv(groups)
    ]
    mean_losses = np.mean(losses, axis=0)
    return RIDGE_ALPHAS[np.argmin(mean_losses, axis=0)]


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    groups_train: np.ndarray,
) -> np.ndarray:
    """Fit ridge with an exact grouped-CV spectral alpha search."""
    alpha = _ridge_alpha(x_train, y_train, groups_train)
    return Ridge(alpha=alpha).fit(x_train, y_train).predict(x_test)


def _ridge_predict_by_target(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    groups_train: np.ndarray,
) -> np.ndarray:
    """Fit independent targets while reusing their spectral CV decompositions."""
    alphas = _ridge_alphas_by_target(x_train, y_train, groups_train)
    return Ridge(alpha=alphas).fit(x_train, y_train).predict(x_test)


def _relabelled_groups(groups: np.ndarray, seed: int) -> np.ndarray:
    """Reorder group tie-breaks without shuffling label counts between groups."""
    unique, codes = np.unique(groups, return_inverse=True)
    return np.random.default_rng(seed).permutation(len(unique))[codes]


def _inner_cv(
    y_train: np.ndarray, groups_train: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified subject-grouped folds for logistic penalty selection."""
    classes = np.unique(y_train)
    support = min(len(np.unique(groups_train[y_train == label])) for label in classes)
    n_splits = min(INNER_CV, support)
    if n_splits < 2:
        raise SubmissionError(
            "an outer training fold has fewer than two subjects in a class"
        )
    rows = np.zeros(len(y_train))
    relabelled_groups = _relabelled_groups(groups_train, RANDOM_STATE)
    splitter = StratifiedGroupKFold(n_splits=n_splits)
    splits = list(splitter.split(rows, y_train, relabelled_groups))
    expected_classes = set(classes)
    if not all(
        set(np.unique(y_train[train])) == expected_classes
        and set(np.unique(y_train[test])) == expected_classes
        for train, test in splits
    ):
        raise SubmissionError(
            "could not construct inner subject-grouped folds with every class"
        )
    return splits


def fit_predict(
    task_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classes: np.ndarray,
    groups_train: np.ndarray,
    inner_cv_jobs: int = INNER_CV_JOBS,
    logreg_tol: float | None = None,
) -> np.ndarray:
    """Fit the fixed linear probe on train, return predictions for test."""
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    logreg_tol = LOGREG_TOL if logreg_tol is None else logreg_tol
    if task_type == "classification":
        if len(np.unique(y_train)) < 2:
            raise SubmissionError("a training fold has a single class")
        scoring = "roc_auc" if len(classes) == 2 else "roc_auc_ovr_weighted"
        clf = LogisticRegressionCV(
            Cs=LOGREG_CS,
            cv=_inner_cv(y_train, groups_train),
            scoring=scoring,
            solver="lbfgs",
            max_iter=MAX_ITER,
            n_jobs=inner_cv_jobs,
            random_state=RANDOM_STATE,
            tol=logreg_tol,
        ).fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        col = {c: i for i, c in enumerate(clf.classes_)}
        out = np.zeros((x_test.shape[0], len(classes)))
        for j, c in enumerate(classes):
            if c in col:
                out[:, j] = proba[:, col[c]]
        return out
    return _ridge_predict(x_train, y_train, x_test, groups_train)


def _fold_score(
    metric: str,
    task_type: str,
    y: np.ndarray,
    folds: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
    inner_cv_jobs: int = INNER_CV_JOBS,
    logreg_tol: float | None = None,
) -> float:
    """Return the mean native metric across one external CV partition."""
    scores = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        predictions = fit_predict(
            task_type,
            matrix[~test],
            y[~test],
            matrix[test],
            classes,
            groups[~test],
            inner_cv_jobs,
            logreg_tol,
        )
        try:
            scores.append(METRICS[metric](y[test], predictions, classes))
        except ValueError as error:
            raise SubmissionError(
                f"CV fold {fold} is not scoreable: {error}"
            ) from error
    if not np.isfinite(scores).all():
        raise SubmissionError("a CV fold produced a non-finite score")
    return float(np.mean(scores))


def _parallel_fold_score(
    metric: str,
    task_type: str,
    y: np.ndarray,
    folds: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
    logreg_tol: float,
) -> float:
    """Score one repeat in an outer worker without nested inner processes."""
    return _fold_score(
        metric,
        task_type,
        y,
        folds,
        matrix,
        classes,
        groups,
        inner_cv_jobs=1,
        logreg_tol=logreg_tol,
    )


def _repeated_cv_scores(
    metric: str,
    task_type: str,
    y: np.ndarray,
    labels: pd.DataFrame,
    matrix: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, ...]:
    """Return one mean fold metric for every external CV repeat."""
    groups = _subject_groups(labels)
    columns = _fold_columns(labels)
    workers = min(OUTER_CV_JOBS, len(columns))
    if workers == 1:
        scores = [
            _fold_score(
                metric,
                task_type,
                y,
                labels[column].to_numpy(),
                matrix,
                classes,
                groups,
            )
            for column in columns
        ]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            scores = Parallel(n_jobs=workers)(
                delayed(_parallel_fold_score)(
                    metric,
                    task_type,
                    y,
                    labels[column].to_numpy(),
                    matrix,
                    classes,
                    groups,
                    LOGREG_TOL,
                )
                for column in columns
            )
    return tuple(float(score) for score in scores)


def _repeated_cv_score(
    metric: str,
    task_type: str,
    y: np.ndarray,
    labels: pd.DataFrame,
    matrix: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Return the aggregate repeated-CV score for legacy callers."""
    return float(
        np.mean(_repeated_cv_scores(metric, task_type, y, labels, matrix, classes))
    )


def _regression_fold_scores(
    metrics: tuple[str, ...],
    targets: np.ndarray,
    folds: np.ndarray,
    matrix: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, ...]:
    """Score compatible regression targets with shared ridge factorizations."""
    fold_scores: list[list[float]] = [[] for _ in metrics]
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        scaler = StandardScaler().fit(matrix[~test])
        x_train = scaler.transform(matrix[~test])
        x_test = scaler.transform(matrix[test])
        predictions = _ridge_predict_by_target(
            x_train, targets[~test], x_test, groups[~test]
        )
        for index, metric in enumerate(metrics):
            fold_scores[index].append(
                METRICS[metric](targets[test, index], predictions[:, index], None)
            )
    if not all(np.isfinite(scores).all() for scores in fold_scores):
        raise SubmissionError("a CV fold produced a non-finite score")
    return tuple(float(np.mean(scores)) for scores in fold_scores)


def _repeated_regression_scores(
    metrics: tuple[str, ...],
    targets: np.ndarray,
    labels: pd.DataFrame,
    matrix: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    """Return repeat scores per target for compatible regression tasks."""
    groups = _subject_groups(labels)
    columns = _fold_columns(labels)
    workers = min(OUTER_CV_JOBS, len(columns))
    if workers == 1:
        by_repeat = [
            _regression_fold_scores(
                metrics, targets, labels[column].to_numpy(), matrix, groups
            )
            for column in columns
        ]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            by_repeat = Parallel(n_jobs=workers)(
                delayed(_regression_fold_scores)(
                    metrics,
                    targets,
                    labels[column].to_numpy(),
                    matrix,
                    groups,
                )
                for column in columns
            )
    return tuple(
        tuple(float(scores[index]) for scores in by_repeat)
        for index in range(len(metrics))
    )


def _score_regression_group(
    tasks: list[dict], labels_by_task: list[pd.DataFrame], emb: pd.DataFrame
) -> list[TaskScore]:
    """Score regression tasks that have identical samples and assignments."""
    labels = labels_by_task[0]
    matrix = _align(labels, emb)
    targets = np.column_stack([frame[LABEL].to_numpy() for frame in labels_by_task])
    metrics = tuple(str(task["metric"]) for task in tasks)
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise EvaluatorError(f"unknown metric(s) for regression tasks: {unknown}")
    if SPLIT in labels:
        split = labels[SPLIT].to_numpy()
        train, test = split == SPLIT_TRAIN, split == SPLIT_TEST
        if not train.any() or not test.any():
            raise EvaluatorError("transfer labels.csv has an empty train or test side")
        scaler = StandardScaler().fit(matrix[train])
        predictions = _ridge_predict_by_target(
            scaler.transform(matrix[train]),
            targets[train],
            scaler.transform(matrix[test]),
            _subject_groups(labels)[train],
        )
        repeat_scores = tuple(
            (METRICS[metric](targets[test, index], predictions[:, index], None),)
            for index, metric in enumerate(metrics)
        )
        n_samples = int(test.sum())
    else:
        repeat_scores = _repeated_regression_scores(metrics, targets, labels, matrix)
        n_samples = len(labels)
    return [
        TaskScore(
            task_id=_norm_id(task[TASK_ID]),
            dataset_id=_norm_id(task[DATASET_ID]),
            category=str(task["category"]),
            metric=metrics[index],
            score=float(np.mean(repeat_scores[index])),
            n_samples=n_samples,
            repeat_scores=repeat_scores[index],
        )
        for index, task in enumerate(tasks)
    ]


def _uses_leave_pair_out(metric: str, y: np.ndarray, groups: np.ndarray) -> bool:
    """Is this a binary AUROC task with fewer than ``MIN_CV_SUBJECTS`` subjects?

    Paired tasks, where a subject carries both labels, stay on CV.
    """
    return (
        metric == "auroc"
        and len(np.unique(y)) == 2
        and len(np.unique(groups)) < MIN_CV_SUBJECTS
        and pd.Series(y).groupby(groups).nunique().max() == 1
    )


def _subject_labels(y: np.ndarray, subjects: np.ndarray) -> pd.Series:
    """Return one label per subject, indexed by subject id."""
    per_subject = pd.Series(y).groupby(subjects)
    if (per_subject.nunique() > 1).any():
        raise EvaluatorError("a subject carries more than one label")
    return per_subject.first()


def _ranked_pairs(scores: np.ndarray, is_positive: np.ndarray) -> tuple[float, int]:
    """Count correctly ordered positive-negative sample pairs, ties as half."""
    gaps = scores[is_positive][:, None] - scores[~is_positive][None, :]
    if not np.isfinite(gaps).all():
        raise SubmissionError("a leave-pair-out fit produced a non-finite score")
    return float((gaps > 0).sum() + 0.5 * (gaps == 0).sum()), int(gaps.size)


def _pair_ranks(
    y: np.ndarray,
    subjects: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    pair: tuple[str, str],
) -> tuple[float, int]:
    """Fit without one subject pair, then count its sample pairs ranked right."""
    positive, negative = pair
    held_out = np.isin(subjects, [positive, negative])
    scores = fit_predict(
        "classification",
        matrix[~held_out],
        y[~held_out],
        matrix[held_out],
        classes,
        subjects[~held_out],
        inner_cv_jobs=1,
    )[:, 1]
    return _ranked_pairs(scores, subjects[held_out] == positive)


def _leave_pair_out_score(
    y: np.ndarray, subjects: np.ndarray, matrix: np.ndarray, classes: np.ndarray
) -> float:
    """Fraction of sample pairs ranked right, each subject pair held out once."""
    per_subject = _subject_labels(y, subjects)
    pairs = list(
        itertools.product(
            per_subject.index[per_subject == classes[1]],
            per_subject.index[per_subject == classes[0]],
        )
    )
    workers = min(OUTER_CV_JOBS, len(pairs))
    if workers == 1:
        counts = [_pair_ranks(y, subjects, matrix, classes, pair) for pair in pairs]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            counts = Parallel(n_jobs=workers)(
                delayed(_pair_ranks)(y, subjects, matrix, classes, pair)
                for pair in pairs
            )
    correct = sum(pair_correct for pair_correct, _ in counts)
    total = sum(pair_total for _, pair_total in counts)
    return correct / total


def _transfer_predict(
    task_type: str,
    y: np.ndarray,
    split: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on the train cohort, predict the test cohort (no pooling)."""
    train, test = split == SPLIT_TRAIN, split == SPLIT_TEST
    if not train.any() or not test.any():
        raise EvaluatorError("transfer labels.csv has an empty train or test side")
    preds = fit_predict(
        task_type,
        matrix[train],
        y[train],
        matrix[test],
        classes,
        groups[train],
    )
    return y[test], preds


def _select_degs(delta_train: np.ndarray, n_top: int) -> np.ndarray:
    """Select the strongest mean absolute responses using training rows only."""
    n_keep = min(int(n_top), delta_train.shape[1])
    if n_keep <= 0:
        raise EvaluatorError("perturbation task has no response genes to select")
    effect = np.abs(np.mean(delta_train, axis=0))
    return np.argsort(effect, kind="stable")[-n_keep:]


def _mean_scores(scores: list[float]) -> float:
    """Average finite fold scores."""
    finite = np.asarray(scores)[np.isfinite(scores)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _perturbation_fold(
    matrix: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    n_top: int,
    target_centered: bool,
) -> float:
    """Fit and score one leakage-safe perturbation fold."""
    if not train.any() or not test.any():
        raise EvaluatorError("perturbation fold has an empty train or test side")
    genes = _select_degs(delta[train], n_top)
    y_train = delta[train][:, genes]
    y_test = delta[test][:, genes]
    scaler = StandardScaler().fit(matrix[train])
    x_train = scaler.transform(matrix[train])
    x_test = np.clip(
        scaler.transform(matrix[test]),
        -PERTURBATION_TEST_Z_CLIP,
        PERTURBATION_TEST_Z_CLIP,
    )
    predictions = _ridge_predict(x_train, y_train, x_test, groups[train])
    if target_centered:
        correlation = compute_target_centered_sample_spearman(y_test, predictions)
    else:
        training_mean = np.mean(y_train, axis=0)
        correlation = compute_residual_sample_spearman(
            y_test, predictions, training_mean
        )
    if not np.isfinite(correlation):
        raise SubmissionError("a perturbation fold has a degenerate Spearman score")
    return float(np.clip(correlation, -1.0, 1.0))


def _perturbation_repeat(
    matrix: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    n_top: int,
) -> float:
    """Score every fold in one perturbation repeat."""
    return _mean_scores(
        [
            _perturbation_fold(matrix, delta, groups, train, test, n_top, False)
            for train, test in partitions
        ]
    )


def _parallel_perturbation_repeats(
    matrix: np.ndarray,
    delta: np.ndarray,
    groups: np.ndarray,
    repeats: list[list[tuple[np.ndarray, np.ndarray]]],
    n_top: int,
) -> tuple[float, ...]:
    """Score independent perturbation repeats across available workers."""
    workers = (
        min(OUTER_CV_JOBS, len(repeats))
        if len(matrix) >= OUTER_PARALLEL_MIN_SAMPLES
        else 1
    )
    if workers == 1:
        scores = [
            _perturbation_repeat(matrix, delta, groups, folds, n_top)
            for folds in repeats
        ]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            scores = Parallel(n_jobs=workers)(
                delayed(_perturbation_repeat)(matrix, delta, groups, folds, n_top)
                for folds in repeats
            )
    return tuple(float(score) for score in scores)


def _paired_perturbation_scores(
    matrix: np.ndarray,
    delta: np.ndarray,
    labels: pd.DataFrame,
    n_top: int,
) -> tuple[float, ...]:
    """Score each frozen subject-grouped CV partition the labels carry."""
    groups = _subject_groups(labels)
    repeats = []
    for column in _fold_columns(labels):
        folds = labels[column].to_numpy()
        repeats.append(
            [(folds != fold, folds == fold) for fold in sorted(np.unique(folds))]
        )
    return _parallel_perturbation_repeats(matrix, delta, groups, repeats, n_top)


def _transfer_perturbation_scores(
    matrix: np.ndarray,
    delta: np.ndarray,
    labels: pd.DataFrame,
    n_top: int,
    target_centered: bool,
) -> tuple[tuple[float, ...], int]:
    """Train every response-decoding component on the source disease only."""
    groups = _subject_groups(labels)
    split = labels[SPLIT].astype(str).to_numpy()
    train, test = split == SPLIT_TRAIN, split == SPLIT_TEST
    metrics = _perturbation_fold(
        matrix, delta, groups, train, test, n_top, target_centered
    )
    return (metrics,), int(test.sum())


def _cartesian_partitions(
    control_ids: np.ndarray, perturbed_ids: np.ndarray
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Build six 3-fold specimen-isolated partitions for a 3x3 response grid."""
    controls = np.unique(control_ids)
    perturbed = np.unique(perturbed_ids)
    if len(controls) != 3 or len(perturbed) != 3:
        raise EvaluatorError(
            "Cartesian perturbation evaluation requires exactly three controls "
            "and three perturbed specimens."
        )
    repeats = []
    for assignment in itertools.permutations(perturbed):
        folds = []
        for control_id, perturbed_id in zip(controls, assignment):
            test = (control_ids == control_id) & (perturbed_ids == perturbed_id)
            train = (control_ids != control_id) & (perturbed_ids != perturbed_id)
            if test.sum() != 1 or train.sum() != 4:
                raise EvaluatorError("Cartesian labels do not form a complete 3x3 grid")
            folds.append((train, test))
        repeats.append(folds)
    return repeats


def _cartesian_perturbation_scores(
    matrix: np.ndarray,
    delta: np.ndarray,
    labels: pd.DataFrame,
    n_top: int,
) -> tuple[float, ...]:
    """Evaluate all control-to-perturbed permutations without specimen leakage."""
    groups = _subject_groups(labels)
    repeats = _cartesian_partitions(
        labels[CONTROL_ID].astype(str).to_numpy(),
        labels[PERTURBED_ID].astype(str).to_numpy(),
    )
    return _parallel_perturbation_repeats(matrix, delta, groups, repeats, n_top)


def _score_perturbation_task(
    task: dict,
    labels: pd.DataFrame,
    emb: pd.DataFrame,
    targets: PerturbationTargets,
) -> TaskScore:
    """Decode one hidden response matrix from submitted baseline embeddings."""
    if len(labels) != len(targets.delta):
        raise EvaluatorError("labels.csv and targets.npz have different row counts")
    matrix = _align(labels, emb)
    n_top = int(task.get("n_top_de_genes", 0))
    pairing = str(task.get(PAIRING, labels.get(PAIRING, pd.Series([PAIRED])).iloc[0]))
    if SPLIT in labels:
        repeat_scores, n_samples = _transfer_perturbation_scores(
            matrix,
            targets.delta,
            labels,
            n_top,
            bool(task.get("target_centered", False)),
        )
    elif pairing == CARTESIAN:
        repeat_scores = _cartesian_perturbation_scores(
            matrix, targets.delta, labels, n_top
        )
        n_samples = len(labels)
    else:
        repeat_scores = _paired_perturbation_scores(
            matrix, targets.delta, labels, n_top
        )
        n_samples = len(labels)
    return TaskScore(
        task_id=_norm_id(task[TASK_ID]),
        dataset_id=_norm_id(task[DATASET_ID]),
        category=str(task["category"]),
        metric=str(task.get("metric", RESIDUAL_SPEARMAN)),
        score=float(np.mean(repeat_scores)),
        n_samples=n_samples,
        repeat_scores=repeat_scores,
    )


def score_task(
    task: dict,
    labels: pd.DataFrame,
    emb: pd.DataFrame,
    targets: PerturbationTargets | None = None,
) -> TaskScore:
    """Score one task: repeated fold-wise CV, leave-pair-out, or a transfer split.

    Private biology (disease, tissue, and target names) never leaks in.
    """
    metric = task["metric"]
    if task.get("task_type") == PERTURBATION:
        if metric not in {CENTERED_SPEARMAN, RESIDUAL_SPEARMAN}:
            raise EvaluatorError(
                f"perturbation task {task.get(TASK_ID)} needs metric "
                f"'{CENTERED_SPEARMAN}'"
            )
        if targets is None:
            raise EvaluatorError("perturbation task is missing targets.npz")
        return _score_perturbation_task(task, labels, emb, targets)
    if metric not in METRICS:
        raise EvaluatorError(f"unknown metric '{metric}' for task {task.get(TASK_ID)}")
    task_type = task["task_type"]
    matrix = _align(labels, emb)
    y = labels[LABEL].to_numpy()
    y_eval = y
    classes = np.unique(y) if task_type == "classification" else np.array([])
    groups = _subject_groups(labels)

    if SPLIT in labels.columns:
        y_eval, preds = _transfer_predict(
            task_type,
            y,
            labels[SPLIT].to_numpy(),
            matrix,
            classes,
            groups,
        )
        try:
            score = METRICS[metric](y_eval, preds, classes)
        except ValueError as error:
            raise SubmissionError(f"transfer task not scoreable: {error}") from error
        repeat_scores = (score,)
    elif _uses_leave_pair_out(metric, y, groups):
        score = _leave_pair_out_score(y, groups, matrix, classes)
        repeat_scores = (score,)
    else:
        repeat_scores = _repeated_cv_scores(
            task_type=task_type,
            metric=metric,
            y=y,
            labels=labels,
            matrix=matrix,
            classes=classes,
        )
        score = float(np.mean(repeat_scores))

    return TaskScore(
        task_id=_norm_id(task[TASK_ID]),
        dataset_id=_norm_id(task[DATASET_ID]),
        category=str(task["category"]),
        metric=metric,
        score=score,
        n_samples=int(len(y_eval)),
        repeat_scores=repeat_scores,
    )


def fetch_manifest(
    token: str | None = None, revision: str = CURRENT.data
) -> list[dict]:
    """Download the public ``datasets.yaml`` registry of opaque dataset ids."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        PUBLIC_REPO,
        MANIFEST_FILENAME,
        repo_type="dataset",
        token=token,
        revision=revision,
    )
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, dict):
        return data.get("datasets", [])
    return data or []


def public_task_ids_from_manifest(path: str | Path) -> set[str]:
    """Read the task ids of the published public-task manifest."""
    with open(path) as handle:
        return {_norm_id(entry[TASK_ID]) for entry in yaml.safe_load(handle)["tasks"]}


def fetch_public_task_ids(
    token: str | None = None, revision: str = CURRENT.data
) -> set[str]:
    """Download the public task ids; none until published or if the fetch fails.

    Losing the Public tags beats blanking every page over a Hub hiccup.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(
            PUBLIC_REPO,
            PUBLIC_TASKS_MANIFEST,
            repo_type="dataset",
            token=token,
            revision=revision,
        )
        return public_task_ids_from_manifest(path)
    except EntryNotFoundError:
        return set()
    except Exception:  # noqa: BLE001
        logging.warning("public task manifest unreadable; no task tagged public")
        return set()


def manifest_ids(manifest: list[dict], modality: str | None = None) -> set[str]:
    """Return manifest dataset ids, optionally for one modality."""
    return {
        _norm_id(entry["id"])
        for entry in manifest
        if modality is None or entry.get("modality") == modality
    }


def scoreable_tasks(tasks: list[dict], valid_ids: set[str]) -> list[dict]:
    """Registry tasks whose dataset is in the public manifest; orphans are skipped.

    Shared by ``score_all`` and the leaderboard so both agree on what full
    coverage means. An orphan (a task on a dataset not yet published) is logged
    and dropped rather than making full coverage unreachable for everyone.
    """
    kept, orphans = [], []
    for task in tasks:
        if _norm_id(task[DATASET_ID]) in valid_ids:
            kept.append(task)
        else:
            orphans.append(_norm_id(task[TASK_ID]))
    if orphans:
        logging.warning("skipping tasks on unpublished datasets: %s", orphans)
    return kept


def fetch_tasks_registry(
    token: str | None = None, revision: str = CURRENT.labels
) -> list[dict]:
    """Download the private ``tasks.yaml`` registry from the labels repo."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        LABELS_REPO, TASKS_FILENAME, repo_type="dataset", token=token, revision=revision
    )
    return load_tasks_registry(path)


def fetch_task_labels(
    task_id: str, token: str | None = None, revision: str = CURRENT.labels
) -> pd.DataFrame:
    """Download a task's private ``labels.csv`` from the labels repo."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        LABELS_REPO,
        f"{task_id}/{LABELS_FILENAME}",
        repo_type="dataset",
        token=token,
        revision=revision,
    )
    return load_labels(path)


def fetch_task_targets(
    task_id: str, token: str | None = None, revision: str = CURRENT.labels
) -> PerturbationTargets:
    """Download a task's private compressed response matrix."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        LABELS_REPO,
        f"{task_id}/{TARGETS_FILENAME}",
        repo_type="dataset",
        token=token,
        revision=revision,
    )
    return load_targets(path)


def _retry(fn):
    """Call ``fn``, retrying transient failures with linear backoff."""
    last: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001
            last = error
            if attempt < FETCH_ATTEMPTS - 1:
                time.sleep(FETCH_BACKOFF * (attempt + 1))
    raise last


def score_all(
    path: str | Path,
    token: str | None = None,
    *,
    modality: str | None = None,
    representation: str = EMBEDDING,
    datasets: list[dict] | None = None,
    tasks: list[dict] | None = None,
    fetch_labels=None,
    fetch_targets=None,
    checkpoint_dir: str | Path | None = None,
    reference: str | Path | None = None,
) -> dict:
    """Score a submission per task and roll it up into a blind result.

    Unknown dataset ids and bad files raise/record a ``SubmissionError``; a
    failed fetch or internal error raises ``EvaluatorError`` (our side, not the
    submitter's). A dataset is embedded once and reused across all its tasks.

    ``EMBEDDING`` columns are put on the unit sphere first, ``GENE_SPACE`` ones
    keep their scale. A ``reference`` file (same layout as a submission, gene
    space) is then appended to every sample and the concatenated row put back
    on the unit sphere.
    """
    if representation not in REPRESENTATIONS:
        raise SubmissionError(
            f"unknown representation '{representation}'; expected one of "
            f"{list(REPRESENTATIONS)}"
        )
    blocks = load_submission(path)
    if datasets is None:
        try:
            datasets = fetch_manifest(token)
        except Exception as error:  # noqa: BLE001
            raise EvaluatorError(
                f"could not load the dataset manifest: {error}"
            ) from error
    if modality is not None and modality not in MODALITIES:
        raise SubmissionError(f"unknown or unavailable modality: {modality}")
    all_ids = manifest_ids(datasets)
    valid = manifest_ids(datasets, MODALITIES.get(modality))
    if modality is not None and not valid:
        raise SubmissionError(f"unknown or unavailable modality: {modality}")

    unknown = sorted(set(blocks) - all_ids)
    if unknown:
        raise SubmissionError(f"unknown dataset_id(s) not in the benchmark: {unknown}")
    wrong_modality = sorted(set(blocks) - valid)
    if wrong_modality:
        raise SubmissionError(
            f"dataset_id(s) outside the selected modality: {wrong_modality}"
        )

    if tasks is None:
        try:
            tasks = fetch_tasks_registry(token)
        except Exception as error:  # noqa: BLE001
            raise EvaluatorError(
                f"could not load the task registry: {error}"
            ) from error
    fetch_labels = fetch_labels or fetch_task_labels
    fetch_targets = fetch_targets or fetch_task_targets
    checkpoints = (
        _TaskCheckpoints(checkpoint_dir, path, modality, representation, reference)
        if checkpoint_dir is not None
        else None
    )
    reference_by_ds = _load_reference(reference) if reference is not None else None

    emb_by_ds, block_error = {}, {}
    for dataset_id, block in blocks.items():
        try:
            frame = _to_embedding_frame(block)
            if representation == EMBEDDING:
                frame = _l2_normalize_rows(frame)
            if reference_by_ds is not None:
                if dataset_id not in reference_by_ds:
                    raise EvaluatorError(
                        f"reference features lack dataset {dataset_id}"
                    )
                frame = _concat_reference(frame, reference_by_ds[dataset_id])
            emb_by_ds[dataset_id] = frame
        except SubmissionError as error:
            block_error[dataset_id] = str(error)

    selected_tasks = [
        task
        for task in scoreable_tasks(tasks, all_ids)
        if _norm_id(task[DATASET_ID]) in valid
    ]
    outcomes: list[TaskOutcome | None] = [None] * len(selected_tasks)
    pending: list[_PendingTask] = []
    for position, task in enumerate(selected_tasks):
        task_id = _norm_id(task[TASK_ID])
        dataset_id = _norm_id(task[DATASET_ID])
        emb = emb_by_ds.get(dataset_id)
        error = block_error.get(dataset_id)
        if emb is None and error is None:
            outcomes[position] = TaskOutcome(task_id, dataset_id, "missing")
            continue
        if error is not None:
            outcomes[position] = TaskOutcome(
                task_id, dataset_id, "invalid", reason=error
            )
            continue
        try:
            labels = _retry(lambda: fetch_labels(task_id, token))
        except Exception as fetch_error:  # noqa: BLE001
            raise EvaluatorError(
                f"could not load evaluation data for a task: {fetch_error}"
            ) from fetch_error
        if reference_by_ds is not None:
            _check_reference_covers(labels, reference_by_ds[dataset_id], task_id)
        targets = None
        if task.get("task_type") == PERTURBATION:
            try:
                targets = _retry(lambda: fetch_targets(task_id, token))
            except Exception as fetch_error:  # noqa: BLE001
                raise EvaluatorError(
                    f"could not load perturbation targets for a task: {fetch_error}"
                ) from fetch_error
        fingerprint = _task_fingerprint(task, labels, targets)
        restored = (
            checkpoints.load(task_id, fingerprint) if checkpoints is not None else None
        )
        if restored is not None:
            outcomes[position] = restored
            continue
        pending.append(_PendingTask(position, task, emb, labels, targets, fingerprint))

    groups: dict[tuple[str, str], list[_PendingTask]] = defaultdict(list)
    for item in pending:
        if item.task.get("task_type") == "regression":
            key = (_norm_id(item.task[DATASET_ID]), _regression_geometry(item.labels))
        else:
            key = ("individual", _norm_id(item.task[TASK_ID]))
        groups[key].append(item)

    for items in groups.values():
        scores = None
        if len(items) > 1 and all(
            item.task.get("task_type") == "regression" for item in items
        ):
            try:
                scores = _score_regression_group(
                    [item.task for item in items],
                    [item.labels for item in items],
                    items[0].emb,
                )
            except SubmissionError:
                scores = None
        for index, item in enumerate(items):
            try:
                score = (
                    scores[index]
                    if scores is not None
                    else score_task(item.task, item.labels, item.emb, item.targets)
                )
                outcome = _outcome_from_score(item.task, score)
            except SubmissionError as scoring_error:
                outcome = TaskOutcome(
                    _norm_id(item.task[TASK_ID]),
                    _norm_id(item.task[DATASET_ID]),
                    "invalid",
                    reason=str(scoring_error),
                )
            outcomes[item.position] = outcome
            if checkpoints is not None:
                checkpoints.save(outcome, item.fingerprint)

    if any(outcome is None for outcome in outcomes):
        raise EvaluatorError("internal error: a task produced no outcome")
    return _summarize([outcome for outcome in outcomes if outcome is not None])


def _summarize(outcomes: list[TaskOutcome]) -> dict:
    """Combine per-task outcomes into blind per-category numbers + coverage."""
    scored = [o.score for o in outcomes if o.status == "scored"]
    dataset_status = _dataset_status(outcomes)
    n_scored, n_total = len(scored), len(outcomes)
    n_ds_scored = sum(1 for status in dataset_status.values() if status == "scored")
    return {
        "per_task": scored,
        "categories": category_means(scored),
        "invalid": [
            {"task_id": o.task_id, "dataset_id": o.dataset_id, "reason": o.reason}
            for o in outcomes
            if o.status == "invalid"
        ],
        "missing": sorted({o.dataset_id for o in outcomes if o.status == "missing"}),
        "incomplete": sorted(
            ds for ds, status in dataset_status.items() if status == "incomplete"
        ),
        "coverage": n_scored / n_total if n_total else 0.0,
        "n_scored": n_scored,
        "n_total": n_total,
        "full_coverage": n_total > 0 and n_scored == n_total,
        "n_datasets_scored": n_ds_scored,
        "n_datasets_total": len(dataset_status),
    }


def _dataset_status(outcomes: list[TaskOutcome]) -> dict[str, str]:
    """Blind per-dataset STATUS only (no score): scored / missing / incomplete."""
    statuses: dict[str, set[str]] = defaultdict(set)
    for outcome in outcomes:
        statuses[outcome.dataset_id].add(outcome.status)
    out = {}
    for dataset_id in sorted(statuses):
        seen = statuses[dataset_id]
        if seen == {"missing"}:
            out[dataset_id] = "missing"
        elif seen == {"scored"}:
            out[dataset_id] = "scored"
        else:
            out[dataset_id] = "incomplete"
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Score a PRIMO submission locally.")
    parser.add_argument("--submission", required=True, help="CSV/TSV/Parquet/NPZ file")
    parser.add_argument("--modality", required=True, choices=MODALITIES)
    parser.add_argument(
        "--representation",
        default=EMBEDDING,
        choices=REPRESENTATIONS,
        help="what the columns are: learned embedding dims, or gene-space values",
    )
    parser.add_argument("--token", default=None, help="HF token (else env HF_TOKEN)")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="resume completed tasks from this directory",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="gene-space features to concatenate to every sample before scoring",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    result = score_all(
        args.submission,
        token,
        modality=args.modality,
        representation=args.representation,
        checkpoint_dir=args.checkpoint_dir,
        reference=args.reference,
    )
    print(
        f"scored     : {result['n_datasets_scored']}/{result['n_datasets_total']} "
        f"datasets, {result['n_scored']}/{result['n_total']} tasks  "
        f"(full_coverage={result['full_coverage']})"
    )
    for category, stats in sorted(result["categories"].items()):
        print(
            f"  {category:20s} {stats['metric']} = {stats['mean']:.4f} "
            f"(n={stats['n_tasks']})"
        )
    for bad in result["invalid"]:
        print(f"  INVALID [{bad['dataset_id']}]: {bad['reason']}")
    if result["missing"]:
        print(f"  missing    : {result['missing']}")
    if result["incomplete"]:
        print(f"  incomplete : {result['incomplete']}")


if __name__ == "__main__":
    _cli()
