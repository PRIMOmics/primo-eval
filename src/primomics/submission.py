"""Read a submission: one embedding per sample, for each dataset."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from primomics.errors import SubmissionError

SAMPLE_ID = "sample_id"
DATASET_ID = "dataset_id"

NPZ_ID_KEYS = ("sample_ids", "sample_id", "ids")
NPZ_EMB_KEYS = ("embeddings", "embedding", "emb", "X")
NPZ_DSID_KEYS = ("dataset_ids", "dataset_id")
NOT_NPZ = "not a .npz archive: save it with np.savez, or use .csv/.parquet"


def norm_id(value: object) -> str:
    """Normalise an id so int/str/float spellings of the same id join."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


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
    except ValueError as err:
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
        dataset_id = norm_id(raw_ds)
        if dataset_id in blocks:
            raise SubmissionError(
                f"dataset_id '{dataset_id}' appears under multiple spellings"
            )
        blocks[dataset_id] = group.drop(columns=[ds_col]).dropna(axis=1, how="all")
    if not blocks:
        raise SubmissionError("empty submission")
    return blocks


def to_embedding_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn one dataset's block into a float frame indexed by sample id."""
    lower = {str(c).lower(): c for c in frame.columns}
    if SAMPLE_ID not in lower:
        raise SubmissionError(
            f"submission needs a '{SAMPLE_ID}' column; found {list(frame.columns)}"
        )
    id_col = lower[SAMPLE_ID]
    ids = [norm_id(v) for v in frame[id_col]]
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


def l2_normalize_rows(emb: pd.DataFrame) -> pd.DataFrame:
    """Put every sample on the unit sphere; an embedding dimension has no unit."""
    norms = np.linalg.norm(emb.to_numpy(dtype=float), axis=1, keepdims=True)
    return pd.DataFrame(
        emb.to_numpy(dtype=float) / np.where(norms == 0.0, 1.0, norms),
        index=emb.index,
        columns=emb.columns,
    )
