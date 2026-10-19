# primo-eval

Evaluation harness, task specs and baselines for the PRIMO benchmark. The Python
package is `primomics`.

Work in progress: nothing here is stable yet.

## Usage

Installing the package adds the `primo` command.

Download the datasets of one modality, `bulk-rna` or `single-cell-rna`:

```bash
primo download --modality bulk-rna
```

The data comes from the
[`PRIMOmics/primo`](https://huggingface.co/datasets/PRIMOmics/primo) dataset on the
Hugging Face Hub. `primo download` always fetches the latest commit, into
`primo-data/` by default (`--data-dir` changes it):

- one `expression.h5ad` per dataset, listed in `datasets.yaml`;
- the labels of the public tasks, under `public_tasks/`.

Use `--public-only` for the datasets that carry a public task, or `--datasets` for a
comma-separated list of ids. `primo-lock.json` records the commit and a hash of
every downloaded file.

Embed the datasets with your model, then score the file:

```bash
primo evaluate submission.parquet --modality bulk-rna -o results.json
```

A submission file has one row per sample: a `dataset_id` column, a `sample_id`
column, then one numeric column per embedding dimension (`.parquet`, `.csv` or
`.tsv`). A `.npz` file holds three arrays instead: `dataset_ids`, `sample_ids` and
`embeddings`. In single-cell datasets, the rows of `expression.h5ad` are cells, and
`obs["sample_id"]` gives their sample: submit one embedding per sample.

`primo evaluate` checks every dataset of the file against the downloaded data: each
sample of `expression.h5ad` needs one finite embedding. It then scores the public
tasks of those datasets with the leaderboard protocol, and lists the other
datasets as validated: the leaderboard scores them. Each embedding is put on the
unit sphere first, as on the leaderboard. The datasets of the modality that are
missing from the file are listed too: on submit, their tasks count as missing.
`results.json` holds every score, with the version of `primomics` and the hashes of
the data files.

## Development

With [uv](https://docs.astral.sh/uv/), which also installs Python 3.12 if needed:

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
```

Before you open a pull request, read `CONTRIBUTING.md`.

## License

Apache-2.0, see `LICENSE`.
