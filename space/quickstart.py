"""Produce a valid PRIMO submission in one command, then swap in your own model.

Downloads one modality's public datasets, at the data commit that the
leaderboard scores, and writes the file the Submit tab expects. The embedding
here is deliberately dumb: bulk data uses log2(CPM+1); single-cell data uses
per-cell log2(CP10K+1), mean pooling by collection sample, then PCA. Replace
``embed`` with your encoder and nothing else changes.

    pip install anndata scikit-learn pandas pyarrow pyyaml huggingface_hub
    python quickstart.py --modality bulk-rna --out submission.parquet

Standalone on purpose: no import from this Space, so it keeps working if you
copy the file into your own project. For the same reason, ``DATA_REVISION``
repeats the data commit of ``versions.yaml``.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from sklearn.decomposition import PCA

PUBLIC_REPO = "PRIMOmics/primo"
DATA_REVISION = "0f452b2fc84e544fb1a1fc5bd42017482e07b43c"
MANIFEST_FILENAME = "datasets.yaml"
MODALITIES = {"bulk-rna": "bulk RNA", "single-cell-rna": "Single Cell RNA"}

DATASET_ID = "dataset_id"
SAMPLE_ID = "sample_id"

TARGET_SUM = 1_000_000
SINGLE_CELL_TARGET_SUM = 10_000
N_COMPONENTS = 50
RANDOM_STATE = 0


def prediction_ids(adata: ad.AnnData, cells: bool) -> np.ndarray:
    """Opaque collection-sample ids required in the submission."""
    if cells:
        return pd.unique(adata.obs[SAMPLE_ID].astype(str)).astype(str)
    return adata.obs_names.astype(str).to_numpy()


def _log_normalize(x, target_sum: int) -> np.ndarray:
    """Return dense log2 counts-per-target expression."""
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    totals = x.sum(axis=1, keepdims=True)
    return np.log2(x / np.where(totals == 0, 1, totals) * target_sum + 1)


def _sample_expression(adata: ad.AnnData, cells: bool) -> tuple[np.ndarray, np.ndarray]:
    """Normalize expression and mean-pool cells into collection samples."""
    ids = prediction_ids(adata, cells)
    if not cells:
        return ids, _log_normalize(adata.X, TARGET_SUM)
    samples = adata.obs[SAMPLE_ID].astype(str).to_numpy()
    pooled = [
        _log_normalize(adata.X[samples == sample_id], SINGLE_CELL_TARGET_SUM).mean(0)
        for sample_id in ids
    ]
    return ids, np.asarray(pooled)


def embed(adata: ad.AnnData, cells: bool) -> np.ndarray:
    """One dataset's expression -> one vector per collection sample. Replace me.

    ``cells`` is true when the rows of ``adata`` are cells, which
    ``obs["sample_id"]`` maps to collection samples.

    ``X`` holds raw counts unless ``adata.uns["input_normalization"]`` names another
    scale (FPKM, microarray intensities or author-normalised counts). The
    log2(CPM + 1) below accepts any non-negative scale.

    Whatever you return, the contract is one finite row per id returned by
    ``prediction_ids``. The embedding width may differ between datasets.
    """
    _, x = _sample_expression(adata, cells)
    k = min(N_COMPONENTS, x.shape[0] - 1, x.shape[1])
    return PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(x)


def download(modality: str, token: str | None) -> Path:
    """Download the manifest and expression files for one modality."""
    manifest = Path(
        hf_hub_download(
            PUBLIC_REPO,
            MANIFEST_FILENAME,
            repo_type="dataset",
            token=token,
            revision=DATA_REVISION,
        )
    )
    paths = [
        MANIFEST_FILENAME,
        *[str(entry["path"]) for entry in datasets(manifest, modality)],
    ]
    return Path(
        snapshot_download(
            PUBLIC_REPO,
            repo_type="dataset",
            token=token,
            revision=DATA_REVISION,
            allow_patterns=paths,
        )
    )


def datasets(manifest_path: Path, modality: str) -> list[dict]:
    """Manifest entries for one modality."""
    manifest = yaml.safe_load(manifest_path.read_text())
    entries = manifest.get("datasets", []) if isinstance(manifest, dict) else manifest
    return [entry for entry in entries if entry["modality"] == modality]


def build(root: Path, modality: str) -> pd.DataFrame:
    """Embed one modality into the frame the Submit tab expects.

    Datasets of different widths stack into one table; the extra columns of a
    narrower dataset stay empty and the evaluator drops them per dataset, so each
    dataset keeps its own embedding size.
    """
    blocks = []
    for entry in datasets(root / MANIFEST_FILENAME, modality):
        dataset_id = str(entry["id"])
        adata = ad.read_h5ad(root / entry["path"])
        cells = entry.get("observation_level") == "cell"
        sample_ids = prediction_ids(adata, cells)
        vectors = embed(adata, cells)
        print(
            f"{dataset_id}: {adata.n_obs} observations, {len(sample_ids)} samples, "
            f"{entry.get('input_normalization', 'raw counts')} "
            f"-> {vectors.shape[1]} dims"
        )
        block = pd.DataFrame(
            vectors, columns=[f"e{i}" for i in range(vectors.shape[1])]
        )
        block.insert(0, SAMPLE_ID, sample_ids)
        block.insert(0, DATASET_ID, dataset_id)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modality", required=True, choices=MODALITIES)
    parser.add_argument("--out", type=Path, default=Path("submission.parquet"))
    parser.add_argument("--token", default=None, help="HF token, if you need one.")
    args = parser.parse_args()

    modality = MODALITIES[args.modality]
    submission = build(download(modality, args.token), modality)
    if args.out.suffix == ".csv":
        submission.to_csv(args.out, index=False)
    else:
        submission.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}: {len(submission)} rows. Upload it on the Submit tab.")


if __name__ == "__main__":
    main()
