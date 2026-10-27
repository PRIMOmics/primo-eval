import anndata
import yaml
from conftest import DATASETS
from quickstart import build, download
from versions import CURRENT

from primomics import data


def test_quickstart_embeds_the_ids_primo_evaluate_expects(local_data):
    path = local_data / "d902" / "expression.h5ad"
    adata = anndata.read_h5ad(path)
    adata.obs["sample_id"] = ["p1"] * 5 + ["p2"] * 5
    adata.write_h5ad(path)

    for entry in DATASETS:
        submission = build(local_data, entry["modality"])
        ids = submission.loc[submission["dataset_id"] == entry["id"], "sample_id"]
        assert ids.tolist() == data.sample_ids(local_data, entry)


def test_quickstart_downloads_the_data_commit_of_the_board_version(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "datasets.yaml"
    manifest.write_text(yaml.safe_dump({"datasets": DATASETS}))
    revisions = []

    def download_manifest(repo_id, filename, **kwargs):
        revisions.append(kwargs["revision"])
        return str(manifest)

    def download_files(repo_id, **kwargs):
        revisions.append(kwargs["revision"])
        return str(tmp_path)

    monkeypatch.setattr("quickstart.hf_hub_download", download_manifest)
    monkeypatch.setattr("quickstart.snapshot_download", download_files)

    assert download("bulk RNA", None) == tmp_path
    assert revisions == [CURRENT.data, CURRENT.data]
