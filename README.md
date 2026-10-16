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

The data comes from the [`PRIMOmics/primo`](https://huggingface.co/datasets/PRIMOmics/primo)
dataset on the Hugging Face Hub. `primo download` always fetches the latest commit,
into `primo-data/` by default (`--data-dir` changes it):

- one `expression.h5ad` per dataset, listed in `datasets.yaml`;
- the labels of the public tasks, under `public_tasks/`.

Use `--public-only` for the datasets that carry a public task, or `--datasets` for a
comma-separated list of ids. `primo-lock.json` records the commit and a hash of
every downloaded file.

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
