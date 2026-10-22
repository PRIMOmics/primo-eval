"""Linear-baseline submissions for the PRIMO leaderboard, such as HVG-1200-genes.

The benchmark scores SUBMITTED EMBEDDINGS: a submitter sends one vector per
sample and the evaluator fits its fixed task probe (linear for scalar targets,
multi-output ridge for perturbation responses) on them. So a "linear baseline" is
not new model code: it is a choice of EMBEDDING. Here the embedding is the
expression itself, reduced to a few features:

    bulk: raw counts -> log2(CPM+1) -> [top-variance genes] -> [PCA]
    single-cell: raw counts -> per-cell log2(CP10K+1) -> sample mean -> [HVG/PCA]

with no gene scaling. Feature extraction is unsupervised (label-blind) and fit
once on each dataset's FULL expression, including the held-out folds. Only the
probe is fit per fold, in ``space/evaluator.py``, so the selection is
transductive: a pipeline picking its genes inside each training fold would see
strictly less.

``n_random`` is the other end of the scale: an embedding drawn from noise, which
ignores the expression entirely. Publishing one anchors the zero of a board.
None is in ``SWEEP`` today.

One config -> one submission file, built from the data that ``primo download``
writes. Anyone can then score its public tasks with ``primo evaluate``:

    uv run python scripts/baselines.py --data-dir primo-data --modality bulk-rna \\
        --out-dir baselines

``--score`` scores every task with the Space scorer, for a local comparison. The
labels come from the Hub with ``HF_TOKEN``, or from a local copy of the labels
repo with ``--labels-dir``.

``--publish`` prepares a new board version: it uploads the HVG-1200 features as
the ``+ HVG`` reference, then logs the data and reference commits to pin in
``space/versions.yaml``. The version rescore scores the ``HVG-1200-genes``
baseline and every ``+ HVG`` entry from them. The features must be built from
the data on the Hub, so ``--publish`` stops when a dataset changed since
``primo download``:

    uv run python scripts/baselines.py --data-dir primo-data --out-dir baselines \\
        --publish
"""

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SPACE_DIR = Path(__file__).resolve().parents[1] / "space"
if str(SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(SPACE_DIR))

from evaluator import (
    DATASET_ID,
    EMBEDDING,
    GENE_SPACE,
    LABELS_FILENAME,
    MANIFEST_FILENAME,
    MODALITIES,
    PUBLIC_REPO,
    RANDOM_STATE,
    SAMPLE_ID,
    TARGETS_FILENAME,
    TASKS_FILENAME,
    fetch_manifest,
    load_labels,
    load_targets,
    load_tasks_registry,
    manifest_ids,
    score_all,
)
from results import HVG_REFERENCE_MODEL, upload_hvg_reference

logger = logging.getLogger(__name__)

TARGET_SUM = 1_000_000
SINGLE_CELL_TARGET_SUM = 10_000

SUBMISSION_PREFIX = "submission"
SCORES_FILENAME = "baselines_scores.csv"


@dataclass(frozen=True)
class BaselineConfig:
    """One linear baseline: optional top-variance gene selection, then optional PCA.

    ``n_hvg`` / ``n_pca`` are upper bounds: both are capped per dataset to what
    its sample and gene counts allow. ``n_random`` replaces the recipe altogether
    with noise of that width, so it cannot be combined with the other two.
    """

    name: str
    n_hvg: int | None = None
    n_pca: int | None = None
    n_random: int | None = None

    def __post_init__(self) -> None:
        if self.n_random is not None and (self.n_hvg or self.n_pca):
            raise ValueError(
                f"Baseline '{self.name}': n_random ignores the expression, so it "
                f"cannot be combined with n_hvg/n_pca."
            )

    @property
    def representation(self) -> str:
        """Gene selection keeps its log2(CPM+1) scale; a projection has no unit."""
        projected = self.n_pca is not None or self.n_random is not None
        return EMBEDDING if projected else GENE_SPACE


SWEEP: tuple[BaselineConfig, ...] = (BaselineConfig("HVG-1200-genes", n_hvg=1200),)
CONFIGS: dict[str, BaselineConfig] = {config.name: config for config in SWEEP}


def log_normalize(x: np.ndarray, target_sum: float) -> np.ndarray:
    """Scale each row to ``target_sum`` counts, then take log2(x + 1)."""
    norm = np.nansum(x, axis=-1, keepdims=True) / target_sum
    norm[norm == 0] = 1
    return np.log1p(x / norm) / math.log(2)


def top_variance_genes(x: np.ndarray, n_genes: int) -> np.ndarray:
    """Column indices of the ``n_genes`` highest-variance genes, in column order.

    Plain variance of the normalized expression. Among genes tied at the cut, the
    higher column index wins.
    """
    order = np.argsort(np.var(x, axis=0), kind="stable")
    return np.sort(order[max(x.shape[1] - n_genes, 0) :])


def prediction_ids(adata: ad.AnnData) -> np.ndarray:
    """Opaque rows required by a submission, in deterministic public-file order."""
    if SAMPLE_ID in adata.obs:
        return pd.unique(adata.obs[SAMPLE_ID].astype(str)).astype(str)
    return adata.obs_names.astype(str).to_numpy()


def _sample_expression(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized sample profiles, pooling cells when needed."""
    ids = prediction_ids(adata)
    if SAMPLE_ID not in adata.obs:
        x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        return ids, log_normalize(x, TARGET_SUM)
    samples = adata.obs[SAMPLE_ID].astype(str).to_numpy()
    pooled = []
    for sample_id in ids:
        x = adata.X[samples == sample_id]
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        pooled.append(log_normalize(x, SINGLE_CELL_TARGET_SUM).mean(axis=0))
    return ids, np.asarray(pooled)


def embed(adata: ad.AnnData, config: BaselineConfig) -> np.ndarray:
    """Embed sample profiles, mean-pooling per-cell CP10K profiles when needed."""
    if config.n_random is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        return rng.standard_normal(
            (len(prediction_ids(adata)), config.n_random), dtype=np.float32
        )
    _, x = _sample_expression(adata)
    if config.n_hvg is not None:
        x = x[:, top_variance_genes(x, config.n_hvg)]
    if config.n_pca is not None:
        k = min(config.n_pca, x.shape[0] - 1, x.shape[1])
        x = PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(x)
    return np.asarray(x, dtype=np.float32)


def load_manifest(data_dir: Path, modality: str | None = None) -> list[dict]:
    """The downloaded ``datasets.yaml``, keeping one modality's datasets if given."""
    from primomics.data import local_manifest

    entries = local_manifest(data_dir)
    if modality is None:
        return entries
    return [entry for entry in entries if entry.get("modality") == MODALITIES[modality]]


def _datasets_to_embed(
    data_dir: Path, modality: str | None, datasets: list[str] | None
) -> list[dict]:
    """The manifest entries to embed, all downloaded, or a ValueError saying why."""
    entries = load_manifest(data_dir, modality)
    if datasets is not None:
        unknown = sorted(set(datasets) - {str(entry["id"]) for entry in entries})
        if unknown:
            raise ValueError(f"not in {MANIFEST_FILENAME} for this modality: {unknown}")
        entries = [entry for entry in entries if str(entry["id"]) in datasets]
    if not entries:
        raise ValueError(f"{data_dir / MANIFEST_FILENAME} lists no dataset to embed.")
    missing = [
        entry["id"] for entry in entries if not (data_dir / entry["path"]).exists()
    ]
    if missing:
        raise ValueError(
            f"not downloaded: {missing}. Run primo download, or pass --datasets with "
            "the datasets you downloaded."
        )
    return entries


def build_submission(
    data_dir: Path,
    config: BaselineConfig,
    modality: str | None = None,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Embed every dataset under one config into a single submission frame.

    Datasets with different embedding widths are aligned into one table; the
    unused columns of a narrower dataset stay NaN and the evaluator drops them
    per dataset, so each dataset keeps its own embedding dimension.
    """
    entries = _datasets_to_embed(data_dir, modality, datasets)
    blocks = []
    for entry in entries:
        dataset_id = str(entry["id"])
        adata = ad.read_h5ad(data_dir / entry["path"])
        sample_ids = prediction_ids(adata)
        emb = embed(adata, config)
        logger.info(
            "[%s] %s: %d samples x %d genes -> %d dims",
            config.name,
            dataset_id,
            len(sample_ids),
            adata.n_vars,
            emb.shape[1],
        )
        block = pd.DataFrame(emb, columns=[f"dim_{i}" for i in range(emb.shape[1])])
        block.insert(0, SAMPLE_ID, sample_ids)
        block.insert(0, DATASET_ID, dataset_id)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def write_submission(
    frame: pd.DataFrame, out_dir: Path, config: BaselineConfig
) -> Path:
    """Write one config's submission to ``submission_<name>.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{SUBMISSION_PREFIX}_{config.name}.parquet"
    frame.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows x %d dims).", path, len(frame), frame.shape[1] - 2)
    return path


def score_submission(
    path: Path,
    data_dir: Path,
    *,
    representation: str,
    labels_dir: Path | None = None,
    modality: str | None = None,
    token: str | None = None,
) -> dict:
    """Score a submission on every task with the Space scorer.

    The labels come from ``labels_dir``, a local copy of the labels repo, when it
    is given, and from the Hub with ``token`` otherwise.
    """
    datasets = load_manifest(data_dir, modality)
    if labels_dir is None:
        return score_all(
            path,
            token,
            modality=modality,
            representation=representation,
            datasets=datasets,
        )

    def fetch_labels(task_id: str, _token: str | None = None) -> pd.DataFrame:
        return load_labels(labels_dir / task_id / LABELS_FILENAME)

    def fetch_targets(task_id: str, _token: str | None = None) -> object:
        return load_targets(labels_dir / task_id / TARGETS_FILENAME)

    return score_all(
        path,
        modality=modality,
        representation=representation,
        datasets=datasets,
        tasks=load_tasks_registry(labels_dir / TASKS_FILENAME),
        fetch_labels=fetch_labels,
        fetch_targets=fetch_targets,
    )


def check_data_is_current(data_dir: Path) -> str:
    """Return the commit of the data on the Hub, or stop if the download differs.

    The reference features must be built from the data that a board version
    pins, so the download has to hold every dataset that the Hub lists at that
    commit, each with the file of the Hub today.
    """
    from huggingface_hub import HfApi

    from primomics.data import changed_datasets

    commit = HfApi().dataset_info(PUBLIC_REPO).sha
    entries = load_manifest(data_dir)
    listed = manifest_ids(fetch_manifest(revision=commit))
    downloaded = manifest_ids(entries)
    if listed != downloaded:
        raise SystemExit(
            "--publish needs the datasets that the Hub lists. Missing from the "
            f"download: {sorted(listed - downloaded)}; no longer on the Hub: "
            f"{sorted(downloaded - listed)}. Run primo download, then publish again."
        )
    changed = changed_datasets(data_dir, entries)
    if changed:
        raise SystemExit(
            f"--publish needs the data of the Hub, and these datasets changed since "
            f"primo download: {changed}. Run primo download, then publish again."
        )
    return commit


def publish_reference(config: BaselineConfig, path: Path, token: str | None) -> str:
    """Upload the features every submission is concatenated with (``+ HVG``).

    Returns the commit of the results repo that holds them, for a new board
    version to pin. The version rescore then scores the baseline from them.
    """
    if config.name != HVG_REFERENCE_MODEL:
        raise ValueError(
            f"only {HVG_REFERENCE_MODEL} is the + HVG reference, not {config.name}"
        )
    commit = upload_hvg_reference(path, token)
    logger.info("Uploaded the %s features as the + HVG reference.", config.name)
    return commit


_META_COLS = ("config", "coverage")


def _score_row(config: BaselineConfig, result: dict) -> dict:
    """Flatten one config's score result into a comparison-table row."""
    row = {"config": config.name, "coverage": result["coverage"]}
    for category, stats in result["categories"].items():
        row[category] = stats["mean"]
    return row


def _log_scores(rows: list[dict]) -> None:
    """Log the sweep, ordered by the hidden mean of the category scores.

    The ranking key (mean across categories) is used to sort but not shown, to
    match the leaderboard.
    """
    categories = sorted({k for row in rows for k in row if k not in _META_COLS})
    frame = pd.DataFrame(rows)
    frame["_rank"] = frame[categories].mean(axis=1)
    frame = frame.sort_values("_rank", ascending=False).drop(columns="_rank")
    logger.info("\n%s", frame.round(4).to_string(index=False))


def _dataset_ids(value: str) -> list[str]:
    """Split a comma-separated list of dataset ids."""
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build (and optionally score and publish) PRIMO linear baselines."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("primo-data"),
        help="The public data, as primo download writes it.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Where to write submissions."
    )
    parser.add_argument(
        "--modality",
        choices=sorted(MODALITIES),
        help="Embed this modality only; omit to embed every dataset.",
    )
    parser.add_argument(
        "--datasets",
        type=_dataset_ids,
        help="Embed only these comma-separated datasets, for a partial download.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        choices=sorted(CONFIGS),
        help="Run these configs; omit to run the full sweep.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score each submission on every task (needs the private labels).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="A local copy of the labels repo; without it, --score reads the Hub.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload the HVG-1200 features as the + HVG reference of a new board version.",
    )
    args = parser.parse_args()

    configs = [CONFIGS[name] for name in args.config] if args.config else list(SWEEP)
    token = os.environ.get("HF_TOKEN")
    if args.publish and not token:
        raise SystemExit("--publish needs write access: set HF_TOKEN.")
    if args.publish and (args.modality or args.datasets):
        raise SystemExit(
            "--publish needs every dataset of every modality: drop --modality and "
            "--datasets."
        )
    if args.publish and HVG_REFERENCE_MODEL not in {config.name for config in configs}:
        raise SystemExit(
            f"--publish uploads the {HVG_REFERENCE_MODEL} features: add it to --config."
        )
    if args.score and args.labels_dir is None and not token:
        raise SystemExit("--score reads the labels on the Hub: set HF_TOKEN.")
    data_commit = check_data_is_current(args.data_dir) if args.publish else None

    rows = []
    for config in configs:
        submission = build_submission(
            args.data_dir, config, args.modality, args.datasets
        )
        path = write_submission(submission, args.out_dir, config)
        if args.score:
            result = score_submission(
                path,
                args.data_dir,
                representation=config.representation,
                labels_dir=args.labels_dir,
                modality=args.modality,
                token=token,
            )
            rows.append(_score_row(config, result))
        if args.publish and config.name == HVG_REFERENCE_MODEL:
            reference_commit = publish_reference(config, path, token)
            logger.info(
                "Pin these commits in space/versions.yaml for the new board version: "
                "data %s, reference %s.",
                data_commit,
                reference_commit,
            )

    if rows:
        _log_scores(rows)
        scores_path = args.out_dir / SCORES_FILENAME
        pd.DataFrame(rows).to_csv(scores_path, index=False)
        logger.info("Wrote %s.", scores_path)


if __name__ == "__main__":
    main()
