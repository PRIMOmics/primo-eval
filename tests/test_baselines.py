import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import baselines
import huggingface_hub
import numpy as np
import pandas as pd
import pytest
import yaml
from baselines import (
    CONFIGS,
    DATASET_ID,
    EMBEDDING,
    GENE_SPACE,
    SAMPLE_ID,
    BaselineConfig,
    build_submission,
    embed,
    prediction_ids,
    score_submission,
)
from scipy.sparse import csr_matrix

import primomics.data


def _adata(x: np.ndarray) -> ad.AnnData:
    obs = pd.DataFrame(
        index=pd.Index([f"s{i:04d}" for i in range(x.shape[0])], name=SAMPLE_ID)
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(x.shape[1])])
    return ad.AnnData(X=x.astype(np.float32), obs=obs, var=var)


def _single_cell_adata() -> ad.AnnData:
    x = csr_matrix(
        np.array(
            [
                [10, 0, 5],
                [0, 20, 0],
                [4, 4, 2],
                [1, 3, 6],
                [5, 0, 5],
                [0, 2, 8],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {SAMPLE_ID: ["s2", "s1", "s2", "s3", "s1", "s3"]},
        index=[f"c{i}" for i in range(6)],
    )
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=["1", "2", "3"]))


def _log2_per_target(x: np.ndarray, target_sum: float) -> np.ndarray:
    return np.log2(x / x.sum(axis=1, keepdims=True) * target_sum + 1)


def _write_data(root: Path, datasets: dict[str, ad.AnnData]) -> None:
    entries = []
    for dataset_id, adata in datasets.items():
        (root / dataset_id).mkdir(parents=True)
        adata.write_h5ad(root / dataset_id / "expression.h5ad")
        modality = "Single Cell RNA" if SAMPLE_ID in adata.obs else "bulk RNA"
        entries.append(
            {
                "id": dataset_id,
                "modality": modality,
                "path": f"{dataset_id}/expression.h5ad",
            }
        )
    (root / "datasets.yaml").write_text(yaml.dump({"datasets": entries}))


def test_hvg_features_match_the_published_baseline():
    """Reference values from the code that built the published HVG-1200 baseline."""
    x = np.array([[1.0, 50, 10, 1000], [1.0, 60, 90, 1000], [1.0, 70, 10, 1000]])
    config = BaselineConfig("hvg-2", n_hvg=2)

    bulk = embed(_adata(x), config)
    single_cell = embed(_single_cell_adata(), config)

    expected_bulk = [
        [15.524246215820312, 13.202441215515137],
        [15.669814109802246, 16.25476837158203],
        [15.982723236083984, 13.175501823425293],
    ]
    expected_single_cell = [
        [12.334554672241211, 11.334844589233398],
        [6.14400053024292, 6.14400053024292],
        [4.983613014221191, 12.758476257324219],
    ]
    np.testing.assert_allclose(bulk, expected_bulk, rtol=1e-6)
    np.testing.assert_allclose(single_cell, expected_single_cell, rtol=1e-6)


def test_embed_raw_is_log2_cpm():
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    emb = embed(_adata(x), BaselineConfig("raw"))
    np.testing.assert_allclose(emb, _log2_per_target(x, 1_000_000), rtol=1e-5)


def test_single_cell_embed_normalizes_cells_and_mean_pools_samples():
    adata = _single_cell_adata()
    emb = embed(adata, BaselineConfig("raw"))
    normalized = _log2_per_target(adata.X.toarray(), 10_000)
    expected = np.vstack(
        [
            normalized[[0, 2]].mean(axis=0),
            normalized[[1, 4]].mean(axis=0),
            normalized[[3, 5]].mean(axis=0),
        ]
    )
    assert prediction_ids(adata).tolist() == ["s2", "s1", "s3"]
    assert emb.shape == (3, 3)
    np.testing.assert_allclose(emb, expected, rtol=1e-5)


def test_a_sample_without_counts_stays_at_zero():
    adata = _adata(np.array([[0.0, 0.0], [1.0, 3.0]]))
    emb = embed(adata, BaselineConfig("raw"))
    np.testing.assert_array_equal(emb[0], [0.0, 0.0])


def test_hvg1200_is_the_only_published_baseline():
    assert set(CONFIGS) == {"HVG-1200-genes"}


def test_the_elo_anchor_names_a_published_baseline():
    """The Elo scale is pinned to a baseline that this script has to keep publishing.

    ``space/`` is a flat tree deployed as-is and cannot import this script, so the
    name is spelled in both; stop spelling it here and every rating silently
    falls back to a mean-centred scale.
    """
    from leaderboard import ELO_ANCHOR_MODEL

    assert ELO_ANCHOR_MODEL in CONFIGS


def test_embed_hvg_keeps_the_top_variance_genes():
    x = np.array([[1.0, 50, 10, 1000], [1.0, 60, 90, 1000], [1.0, 70, 10, 1000]])
    emb = embed(_adata(x), BaselineConfig("hvg-2", n_hvg=2))
    expected = _log2_per_target(x, 1_000_000)
    order = np.argsort(np.var(expected, axis=0))
    np.testing.assert_allclose(emb, expected[:, np.sort(order[-2:])], rtol=1e-5)


def test_tied_genes_at_the_cut_keep_the_higher_column():
    x = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    assert baselines.top_variance_genes(x, 2).tolist() == [1, 2]


def test_embed_pca_reduces_dim():
    rng = np.random.default_rng(0)
    x = rng.poisson(5, size=(30, 40)).astype(float)
    emb = embed(_adata(x), BaselineConfig("pca-10", n_pca=10))
    assert emb.shape == (30, 10)


def test_embed_pca_caps_at_n_samples_minus_one():
    rng = np.random.default_rng(0)
    x = rng.poisson(5, size=(8, 40)).astype(float)
    emb = embed(_adata(x), BaselineConfig("pca-50", n_pca=50))
    assert emb.shape == (8, 7)


def test_embed_hvg_then_pca():
    rng = np.random.default_rng(0)
    x = rng.poisson(5, size=(30, 100)).astype(float)
    emb = embed(_adata(x), BaselineConfig("hvg20-pca5", n_hvg=20, n_pca=5))
    assert emb.shape == (30, 5)


def test_random_embedding_ignores_the_expression():
    """The zero anchor must not read the counts, or it is not an anchor."""
    config = BaselineConfig("random-8", n_random=8)
    emb = embed(_adata(np.ones((12, 40))), config)
    other = embed(_adata(np.full((12, 40), 999.0)), config)
    assert emb.shape == (12, 8)
    np.testing.assert_array_equal(emb, other)


def test_random_cannot_be_mixed_with_a_real_recipe():
    with pytest.raises(ValueError, match="n_random"):
        BaselineConfig("bad", n_pca=10, n_random=8)


def test_gene_recipes_keep_their_scale_and_projections_do_not():
    """A PCA axis has no more unit than a learned dimension; a gene column does."""
    assert BaselineConfig("hvg", n_hvg=1200).representation == GENE_SPACE
    assert BaselineConfig("raw").representation == GENE_SPACE
    assert BaselineConfig("pca", n_pca=64).representation == EMBEDDING
    assert BaselineConfig("noise", n_random=32).representation == EMBEDDING


def test_build_submission_has_ids_and_matches_samples(tmp_path):
    rng = np.random.default_rng(0)
    _write_data(
        tmp_path,
        {
            "d901": _adata(rng.poisson(5, size=(6, 4)).astype(float)),
            "d902": _adata(rng.poisson(5, size=(5, 4)).astype(float)),
        },
    )
    submission = build_submission(tmp_path, BaselineConfig("pca-3", n_pca=3))
    assert list(submission.columns[:2]) == [DATASET_ID, SAMPLE_ID]
    assert len(submission) == 11
    d901 = submission[submission[DATASET_ID] == "d901"]
    assert list(d901[SAMPLE_ID]) == [f"s{i:04d}" for i in range(6)]


def test_build_submission_nan_pads_uneven_dims(tmp_path):
    rng = np.random.default_rng(0)
    _write_data(
        tmp_path,
        {
            "d901": _adata(rng.poisson(5, size=(4, 3)).astype(float)),
            "d902": _adata(rng.poisson(5, size=(4, 5)).astype(float)),
        },
    )
    submission = build_submission(tmp_path, BaselineConfig("raw"))
    d901 = submission[submission[DATASET_ID] == "d901"]
    d902 = submission[submission[DATASET_ID] == "d902"]
    assert d901["dim_4"].isna().all()
    assert d902["dim_4"].notna().all()


def test_build_submission_emits_unique_single_cell_samples(tmp_path):
    _write_data(tmp_path, {"d917": _single_cell_adata()})
    submission = build_submission(tmp_path, BaselineConfig("raw"))
    assert submission[SAMPLE_ID].tolist() == ["s2", "s1", "s3"]
    assert np.isfinite(submission.filter(like="dim_").to_numpy()).all()


def test_build_submission_keeps_one_modality(tmp_path):
    rng = np.random.default_rng(0)
    _write_data(
        tmp_path,
        {
            "d901": _adata(rng.poisson(5, size=(4, 3)).astype(float)),
            "d917": _single_cell_adata(),
        },
    )
    bulk = build_submission(tmp_path, BaselineConfig("raw"), "bulk-rna")
    single_cell = build_submission(tmp_path, BaselineConfig("raw"), "single-cell-rna")
    assert set(bulk[DATASET_ID]) == {"d901"}
    assert set(single_cell[DATASET_ID]) == {"d917"}


def test_build_submission_of_an_empty_modality_fails(tmp_path):
    _write_data(tmp_path, {"d917": _single_cell_adata()})
    with pytest.raises(ValueError, match="no dataset"):
        build_submission(tmp_path, BaselineConfig("raw"), "bulk-rna")


def test_build_submission_of_a_partial_download_names_the_missing_datasets(
    tmp_path,
):
    rng = np.random.default_rng(0)
    _write_data(
        tmp_path,
        {
            "d901": _adata(rng.poisson(5, size=(4, 3)).astype(float)),
            "d902": _adata(rng.poisson(5, size=(4, 3)).astype(float)),
        },
    )
    (tmp_path / "d902" / "expression.h5ad").unlink()

    with pytest.raises(ValueError, match=r"not downloaded: \['d902'\]"):
        build_submission(tmp_path, BaselineConfig("raw"))
    submission = build_submission(tmp_path, BaselineConfig("raw"), datasets=["d901"])
    assert set(submission[DATASET_ID]) == {"d901"}


def test_build_submission_rejects_an_unknown_dataset(tmp_path):
    _write_data(tmp_path, {"d917": _single_cell_adata()})
    with pytest.raises(ValueError, match="d999"):
        build_submission(tmp_path, BaselineConfig("raw"), datasets=["d999"])


def test_score_submission_reads_the_labels_of_a_local_copy(tmp_path, monkeypatch):
    _write_data(tmp_path / "data", {"d917": _single_cell_adata()})
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "tasks.yaml").write_text(
        yaml.dump({"tasks": [{"task_id": "t917", "dataset_id": "d917"}]})
    )
    calls = {}
    monkeypatch.setattr(baselines, "score_all", lambda *a, **kw: calls.update(kw))
    monkeypatch.setattr(baselines, "load_labels", lambda path: path)

    score_submission(
        tmp_path / "sub.parquet",
        tmp_path / "data",
        representation=GENE_SPACE,
        labels_dir=labels_dir,
    )

    assert calls["representation"] == GENE_SPACE
    assert [task["task_id"] for task in calls["tasks"]] == ["t917"]
    assert [entry["id"] for entry in calls["datasets"]] == ["d917"]
    assert calls["fetch_labels"]("t917") == labels_dir / "t917" / "labels.csv"


def test_score_submission_reads_the_hub_without_a_local_copy(tmp_path, monkeypatch):
    _write_data(tmp_path, {"d917": _single_cell_adata()})
    calls = {}
    monkeypatch.setattr(
        baselines, "score_all", lambda path, token, **kw: calls.update(kw, token=token)
    )

    score_submission(
        tmp_path / "sub.parquet",
        tmp_path,
        representation=GENE_SPACE,
        token="hf_test",
    )

    assert calls["token"] == "hf_test"
    assert "tasks" not in calls
    assert "fetch_labels" not in calls


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["baselines.py", *argv])
    baselines.main()


def test_publish_needs_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        _main(
            monkeypatch,
            "--data-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--publish",
        )


@pytest.mark.parametrize(
    "partial", [("--modality", "bulk-rna"), ("--datasets", "d901")]
)
def test_publish_refuses_a_partial_reference(tmp_path, monkeypatch, partial):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    with pytest.raises(SystemExit, match="every dataset of every modality"):
        _main(
            monkeypatch,
            "--data-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--publish",
            *partial,
        )


def test_publish_stops_when_the_data_changed_on_the_hub(tmp_path, monkeypatch):
    _write_data(tmp_path / "data", {"d901": _adata(np.ones((4, 3)))})
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        primomics.data, "changed_datasets", lambda data_dir, entries: ["d901"]
    )
    uploads = []
    monkeypatch.setattr(
        baselines, "upload_hvg_reference", lambda path, token: uploads.append(path)
    )

    with pytest.raises(SystemExit, match=r"changed since primo download: \['d901'\]"):
        _main(
            monkeypatch,
            "--data-dir",
            str(tmp_path / "data"),
            "--out-dir",
            str(tmp_path / "out"),
            "--publish",
        )
    assert uploads == []


def test_publish_uploads_only_the_reference_and_logs_the_commits_to_pin(
    tmp_path, monkeypatch, caplog
):
    rng = np.random.default_rng(0)
    _write_data(tmp_path / "data", {"d901": _adata(rng.poisson(5, size=(4, 3)))})
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        primomics.data, "changed_datasets", lambda data_dir, entries: []
    )
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "dataset_info",
        lambda self, repo: SimpleNamespace(sha="data123"),
    )
    uploads = []
    monkeypatch.setattr(
        baselines,
        "upload_hvg_reference",
        lambda path, token: uploads.append(path) or "ref456",
    )
    monkeypatch.setattr(
        baselines, "score_all", lambda *args, **kwargs: pytest.fail("scored")
    )

    with caplog.at_level(logging.INFO):
        _main(
            monkeypatch,
            "--data-dir",
            str(tmp_path / "data"),
            "--out-dir",
            str(tmp_path / "out"),
            "--publish",
        )

    assert uploads == [tmp_path / "out" / "submission_HVG-1200-genes.parquet"]
    assert "data data123, reference ref456" in caplog.text
