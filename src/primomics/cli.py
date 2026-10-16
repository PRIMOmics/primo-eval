"""The ``primo`` command line.

``primo download`` fetches the datasets of one modality from the Hub, with the
labels of its public tasks. Exit codes: 0 when the command succeeds, 1 when the
input must be fixed (``SubmissionError``), 2 when the failure is on our side
(``EvaluatorError`` or an unexpected error).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import NoReturn

from primomics.data import DEFAULT_DATA_DIR, Download, download
from primomics.errors import EvaluatorError, SubmissionError
from primomics.evaluate import MODALITIES
from primomics.submission import norm_id
from primomics.tasks import PUBLIC_TASKS, load_tasks_registry

logger = logging.getLogger(__name__)


class _Parser(argparse.ArgumentParser):
    """An argparse parser whose usage errors exit with 1: the input must be fixed."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _ids(value: str | None) -> list[str] | None:
    """Split a comma-separated list of ids, or keep ``None``."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _size(n_bytes: int) -> str:
    """A byte count in MB, or in GB from 1 GB."""
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.1f} GB"
    return f"{n_bytes / 1e6:.1f} MB"


def _log_download(result: Download, data_dir: str) -> None:
    """One line per dataset: its samples and its public tasks with labels."""
    logger.info(
        "%d dataset(s), %s, up to date in %s (revision %s).",
        len(result.datasets),
        _size(result.n_bytes),
        data_dir,
        result.revision[:8],
    )
    logger.info("%-10s %8s  %s", "dataset", "samples", "public tasks")
    for entry in result.datasets:
        tasks = result.labelled.get(norm_id(entry["id"]), [])
        logger.info(
            "%-10s %8s  %s",
            entry["id"],
            entry.get("n_samples", ""),
            ", ".join(tasks) or "-",
        )


def _download(args: argparse.Namespace) -> None:
    """Run ``primo download``."""
    result = download(
        Path(args.data_dir),
        args.modality,
        tasks=load_tasks_registry(PUBLIC_TASKS),
        public_only=args.public_only,
        dataset_ids=_ids(args.datasets),
    )
    _log_download(result, args.data_dir)


def _parser() -> argparse.ArgumentParser:
    """The ``primo`` parser, one subcommand per action."""
    parser = _Parser(
        prog="primo", description="Download the PRIMO data and score embeddings."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "download",
        help="download the datasets of one modality",
        description=(
            "Download the latest datasets of one modality from the Hub, with the "
            "labels of its public tasks."
        ),
    )
    fetch.add_argument("--modality", required=True, choices=sorted(MODALITIES))
    fetch.add_argument(
        "--public-only",
        action="store_true",
        help="only the datasets that carry a public task",
    )
    fetch.add_argument("--datasets", help="comma-separated dataset ids")
    fetch.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"where to put the data (default: {DEFAULT_DATA_DIR})",
    )
    fetch.set_defaults(run=_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one ``primo`` command and return its exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(format="%(message)s")
    logging.getLogger("primomics").setLevel(logging.INFO)
    try:
        args.run(args)
    except SubmissionError as error:
        logger.error("error: %s", error)
        return 1
    except EvaluatorError as error:
        logger.error("error on our side: %s", error)
        return 2
    except Exception:
        logger.exception("unexpected error on our side")
        return 2
    return 0
