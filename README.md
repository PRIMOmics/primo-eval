# primo-eval

Evaluation harness, task specs and baselines for the PRIMO benchmark. The Python
package is `primomics`.

Work in progress: nothing here is stable yet.

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
