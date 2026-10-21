# 🧬 primo-eval

Evaluation harness, task specs and baselines for the PRIMO benchmark. The Python
package is `primomics`.

🏆 [Leaderboard](https://huggingface.co/spaces/PRIMOmics/primo-eval) ·
📦 [Data](https://huggingface.co/datasets/PRIMOmics/primo)

🚧 Work in progress: nothing here is stable yet.

## 🚀 From install to the leaderboard

1. ⚡ [Install uv](#-1-install-uv)
2. 📦 [Clone and install primo](#-2-clone-and-install-primo)
3. 📥 [Download the data](#-3-download-the-data)
4. 🧠 [Embed the datasets](#-4-embed-the-datasets)
5. 📊 [Score the public tasks locally](#-5-score-the-public-tasks-locally) (optional)
6. 🏆 [Submit to the leaderboard](#-6-submit-to-the-leaderboard)

You work in two folders, side by side. The clone of this repo runs `primo`, in its
own environment. Your repo runs your model, in your environment:

```text
work/
├── primo-eval/   this repo: download, score and submit
└── my-model/     your repo: embed the data with your model
```

### ⚡ 1. Install uv

The steps below use [uv](https://docs.astral.sh/uv/). It installs Python and the exact
packages of this repo in one command. `primo` supports macOS and Linux. Install uv
with the [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The same page gives other ways, such as `brew install uv`. Open a new terminal, then
check that uv works:

```bash
uv --version
```

### 📦 2. Clone and install primo

```bash
git clone https://github.com/PRIMOmics/primo-eval
cd primo-eval
uv sync
uv run primo --help
```

`uv sync` creates the `.venv` folder, with Python 3.12 and the exact packages of
`uv.lock`. If Python 3.12 is not on your computer, uv downloads it. `uv run` runs a
command in this environment, so you never need to activate it. Run every `primo`
command from this folder.

> [!NOTE]
> `main` can change scores. For stable scores, use a
> [release](https://github.com/PRIMOmics/primo-eval/releases): run
> `git checkout <tag>` before `uv sync`.

### 📥 3. Download the data

Download the datasets of one modality, `bulk-rna` or `single-cell-rna`:

```bash
uv run primo download --modality bulk-rna
```

The data comes from the
[`PRIMOmics/primo`](https://huggingface.co/datasets/PRIMOmics/primo) dataset on the
Hugging Face Hub. It is public, so you need no account. `primo download` always
fetches the latest commit, into `primo-data/` by default (`--data-dir` changes it):

- `datasets.yaml` lists every dataset, with its `id`, `modality` and `path`;
- one `expression.h5ad` per dataset;
- the labels of the public tasks, under `public_tasks/`;
- `primo-lock.json` records the commit and a hash of every downloaded file.

Use `--public-only` for the datasets that carry a public task, or `--datasets` for a
comma-separated list of ids. Run the command again to get the latest data.

### 🧠 4. Embed the datasets

Embed the data in your own repo, with your model and your environment. Write all
the embeddings of one modality to **one** file. The file has one row per sample: a
`dataset_id` column, a `sample_id` column, then one numeric column per embedding
dimension (`.parquet`, `.csv` or `.tsv`). A `.npz` file holds three arrays instead:
`dataset_ids`, `sample_ids` and `embeddings`. Each dataset can have its own number of
dimensions: leave the extra columns empty.

In bulk datasets, each row of `expression.h5ad` is a sample. In single-cell datasets,
the rows are cells, and `obs["sample_id"]` gives their sample: submit one embedding
per sample. `X` holds raw counts, unless the `input_normalization` of the dataset in
`datasets.yaml` gives another scale.

This example embeds every downloaded bulk dataset with a simple baseline:
log2(CPM + 1), then PCA. It is adapted from the
[`quickstart.py`](https://huggingface.co/spaces/PRIMOmics/primo-eval/blob/main/quickstart.py)
of the leaderboard Space. In your environment, it needs anndata, pandas, pyarrow,
pyyaml and scikit-learn. Save it as `embed.py` in your repo, then run it from there:

```python
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

DATA_DIR = Path("../primo-eval/primo-data")
MODALITY = "bulk RNA"


def log_normalize(x, target_sum: float) -> np.ndarray:
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    totals = x.sum(axis=1, keepdims=True)
    return np.log2(x / np.where(totals == 0, 1, totals) * target_sum + 1)


def embed(adata: ad.AnnData, ids, cells: bool) -> np.ndarray:
    """One row per id, in the same order. Replace this baseline with your model."""
    if cells:
        samples = adata.obs["sample_id"].astype(str).to_numpy()
        x = np.stack([log_normalize(adata.X[samples == i], 1e4).mean(0) for i in ids])
    else:
        x = log_normalize(adata.X, 1e6)
    k = min(50, x.shape[0] - 1, x.shape[1])
    return PCA(n_components=k, random_state=0).fit_transform(x)


blocks = []
for entry in yaml.safe_load((DATA_DIR / "datasets.yaml").read_text())["datasets"]:
    path = DATA_DIR / entry["path"]
    if entry["modality"] != MODALITY or not path.exists():
        continue
    adata = ad.read_h5ad(path)
    cells = entry.get("observation_level") == "cell"
    ids = pd.unique(adata.obs["sample_id"].astype(str)) if cells else adata.obs_names
    block = pd.DataFrame(embed(adata, ids, cells)).add_prefix("e")
    block.insert(0, "sample_id", ids)
    block.insert(0, "dataset_id", entry["id"])
    blocks.append(block)
pd.concat(blocks).to_parquet("submission.parquet", index=False)
```

Replace `embed` with your model: it gets one dataset and returns one row per id, in
the order of `ids`. For single-cell data, set `MODALITY = "Single Cell RNA"`, and use
`--modality single-cell-rna` in steps 5 and 6. The baseline then averages the cells
of each sample.

### 📊 5. Score the public tasks locally

From the clone:

```bash
uv run primo evaluate ../my-model/submission.parquet --modality bulk-rna \
  -o ../my-model/results.json
```

`primo evaluate` checks every dataset of the file against the downloaded data: each
sample of `expression.h5ad` needs one finite embedding. It then scores the public
tasks of those datasets with the leaderboard protocol, and lists the other datasets
as validated: the leaderboard scores them. Each embedding is put on the unit sphere
first, as on the leaderboard. The datasets of the modality that are missing from the
file are listed too: on submit, their tasks count as missing. `results.json` holds
every score, with the version of `primomics` and the hashes of the data files.

This step is optional, and it needs no account. It finds the errors in your file
before you submit: when a check fails, the command says what to fix and exits with
code 1.

### 🏆 6. Submit to the leaderboard

#### 💻 From the command line

Log in to Hugging Face once, with an access token from your
[settings](https://huggingface.co/settings/tokens). On a server, you can set the
`HF_TOKEN` environment variable instead. Then, from the clone, check the file and
send it:

```bash
uv run hf auth login
uv run primo submit ../my-model/submission.parquet --modality bulk-rna \
  --model-name my-model --email me@lab.org --dry-run
uv run primo submit ../my-model/submission.parquet --modality bulk-rna \
  --model-name my-model --email me@lab.org
```

`primo submit` runs the checks of `primo evaluate`, and checks that each dataset of
the file is still the latest one on the Hub. If a dataset changed, run
`uv run primo download` again, then embed that dataset again. `--dry-run` runs every
check and sends nothing. Without it, the command sends the file to the
[leaderboard Space](https://huggingface.co/spaces/PRIMOmics/primo-eval), which scores
every task, the private ones included, and publishes the scores. Scoring takes a few
minutes. The Space reads your Hugging Face token only to know who submits: a model
name stays with the account that first used it. `--institution`, `--authors`,
`--paper-link`, `--hf-model` and `--notes` fill the other fields of the web form.

#### 🌐 In the browser

Open the
[**Submit** tab](https://huggingface.co/spaces/PRIMOmics/primo-eval?tab=submit) of the
leaderboard Space, and sign in with Hugging Face. Choose the modality, enter a model
name and a contact email, and upload the file. The other fields are optional. Then
click **Evaluate**: after a few minutes, your scores show on the boards.

The Submit tab also gives the rules: partial submissions, new submissions under the
same name, and the `+ HVG` entry that comes with each submission.

## 🛠️ Development

In the clone (step 2), run the same checks as the CI:

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
```

The leaderboard Space lives in `space/`. A merge does not deploy it: maintainers
deploy it by hand (see `CONTRIBUTING.md`). Its tests run in a separate environment,
with the packages of the Space:

```bash
uv run --isolated --no-project --python 3.13 \
  --with-requirements space/requirements.txt \
  --with "gradio[oauth]==5.50.0" --with pytest \
  pytest space_tests
```

Before you open a pull request, read `CONTRIBUTING.md`.

## 📄 License

Apache-2.0, see `LICENSE`.
