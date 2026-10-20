import anndata
from conftest import DATASETS
from quickstart import build

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
