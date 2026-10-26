"""The ``primo`` command line.

``primo download`` fetches the datasets of one modality from the Hub, at the
commit that the leaderboard scores, with the labels of its public tasks.
``primo evaluate`` checks a submission file against that data, then scores the
public tasks it covers with the leaderboard protocol.
``primo submit`` checks the file the same way, then sends it to the leaderboard
Space, which scores every task. Exit codes: 0 when the command succeeds, 1 when
the input must be fixed (``SubmissionError``), 2 when the failure is on our side
(``EvaluatorError`` or an unexpected error).
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import numpy as np
from huggingface_hub import get_token

from primomics import __version__
from primomics.data import (
    DATA_REPO,
    DATA_REVISION,
    DEFAULT_DATA_DIR,
    OK,
    Download,
    changed_datasets,
    check_submission,
    download,
    labelled_tasks,
    local_fetchers,
    local_manifest,
    locked_files,
)
from primomics.errors import EvaluatorError, SubmissionError
from primomics.evaluate import (
    EMBEDDING,
    MODALITIES,
    REPRESENTATIONS,
    TASK_ID,
    manifest_ids,
    score_all,
)
from primomics.submission import DATASET_ID, norm_id
from primomics.submit import PUBLISHED, REFUSED, SPACE, submit
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


def _sha256(path: str | Path) -> str:
    """The sha256 of a file."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
        revision=args.revision,
    )
    _log_download(result, args.data_dir)


def _public_tasks(manifest: list[dict], modality: str) -> list[dict]:
    """The public tasks on the datasets of one modality."""
    in_modality = manifest_ids(manifest, MODALITIES[modality])
    return [
        task
        for task in load_tasks_registry(PUBLIC_TASKS)
        if norm_id(task[DATASET_ID]) in in_modality
    ]


def _log_checks(verdicts: dict[str, str]) -> None:
    """One line per dataset of the file: ``ok``, or what to fix."""
    logger.info("Checked %d dataset(s):", len(verdicts))
    for dataset_id, verdict in sorted(verdicts.items()):
        logger.info("%-10s %s", dataset_id, verdict)


def _sd(repeat_scores: tuple[float, ...]) -> str:
    """The spread over CV repeats, or ``-`` for a single repeat."""
    if len(repeat_scores) < 2:
        return "-"
    return f"{np.std(repeat_scores, ddof=1):.3f}"


def _log_scores(result: dict, verdicts: dict[str, str], scored: list[dict]) -> None:
    """The score of each public task, the family means, and what was not scored."""
    logger.info(
        "Scored %d of %d labelled public task(s).",
        result["n_scored"],
        result["n_total"],
    )
    if result["per_task"]:
        logger.info("%-8s %-18s %7s %7s %6s", "task", "metric", "score", "sd", "n")
    for score in result["per_task"]:
        logger.info(
            "%-8s %-18s %7.3f %7s %6d",
            score.task_id,
            score.metric,
            score.score,
            _sd(score.repeat_scores),
            score.n_samples,
        )
    for category, stats in sorted(result["categories"].items()):
        logger.info(
            "%s (%s): %.3f over %d task(s)",
            category,
            stats["metric"],
            stats["mean"],
            stats["n_tasks"],
        )
    for item in result["invalid"]:
        logger.warning("not scored: %s (%s)", item["task_id"], item["reason"])
    with_labels = {norm_id(task[DATASET_ID]) for task in scored}
    validated = sorted(
        dataset_id for dataset_id in verdicts if dataset_id not in with_labels
    )
    if validated:
        logger.info("Validated, scored on submit: %s", ", ".join(validated))


def _report(
    args: argparse.Namespace,
    manifest: list[dict],
    verdicts: dict[str, str],
    scored: list[dict],
    unlabelled: list[str],
    result: dict,
) -> dict:
    """The ``results.json`` of one evaluation, with the data it was computed on."""
    in_file = [entry for entry in manifest if norm_id(entry["id"]) in verdicts]
    used_tasks = [
        task[TASK_ID] for task in scored if norm_id(task[DATASET_ID]) in verdicts
    ]
    return {
        "primomics_version": __version__,
        "modality": args.modality,
        "representation": args.representation,
        "submission": {
            "file": Path(args.submission).name,
            "sha256": _sha256(args.submission),
        },
        "data": {
            "repo": DATA_REPO,
            "files": locked_files(args.data_dir, in_file, used_tasks),
        },
        "checks": verdicts,
        "per_task": [asdict(score) for score in result["per_task"]],
        "categories": result["categories"],
        "coverage": result["coverage"],
        "n_scored": result["n_scored"],
        "n_total": result["n_total"],
        "invalid": result["invalid"],
        "missing": result["missing"],
        "unlabelled": unlabelled,
    }


def _checked(args: argparse.Namespace) -> tuple[dict[str, str], list[dict]]:
    """Check the submission file: stop on a failed dataset, warn about absent ones."""
    verdicts = check_submission(args.submission, args.data_dir, args.modality)
    _log_checks(verdicts)
    failed = sorted(
        dataset_id for dataset_id, verdict in verdicts.items() if verdict != OK
    )
    if failed:
        raise SubmissionError(f"fix these datasets, then run again: {failed}")
    manifest = local_manifest(args.data_dir)
    absent = sorted(manifest_ids(manifest, MODALITIES[args.modality]) - set(verdicts))
    if absent:
        logger.warning(
            "missing from the file, their tasks count as missing on submit: %s",
            ", ".join(absent),
        )
    return verdicts, manifest


def _evaluate(args: argparse.Namespace) -> None:
    """Run ``primo evaluate``."""
    if args.output and not Path(args.output).parent.is_dir():
        raise SubmissionError(f"cannot write {args.output}: its folder does not exist")
    verdicts, manifest = _checked(args)
    tasks = _public_tasks(manifest, args.modality)
    scored = labelled_tasks(args.data_dir, tasks)
    unlabelled = [
        task[TASK_ID]
        for task in tasks
        if task not in scored and norm_id(task[DATASET_ID]) in verdicts
    ]
    if unlabelled:
        logger.warning("no local labels, not scored: %s", unlabelled)
    fetch_labels, fetch_targets = local_fetchers(args.data_dir)
    result = score_all(
        args.submission,
        datasets=manifest,
        tasks=scored,
        fetch_labels=fetch_labels,
        fetch_targets=fetch_targets,
        modality=args.modality,
        representation=args.representation,
    )
    _log_scores(result, verdicts, scored)
    if args.output:
        report = _report(args, manifest, verdicts, scored, unlabelled, result)
        Path(args.output).write_text(json.dumps(report, indent=2))
        logger.info("Wrote %s.", args.output)


def _submit(args: argparse.Namespace) -> None:
    """Run ``primo submit``."""
    verdicts, manifest = _checked(args)
    in_file = [entry for entry in manifest if norm_id(entry["id"]) in verdicts]
    changed = changed_datasets(args.data_dir, in_file)
    if changed:
        raise SubmissionError(
            f"the data of {changed} changed on the Hub since your download: run "
            "primo download, then embed these datasets again"
        )
    token = get_token()
    if not token:
        raise SubmissionError("no Hugging Face token: run `hf auth login` first")
    if args.dry_run:
        logger.info("Dry run: %s is ready to submit.", args.submission)
        return
    logger.info(
        "Submitting %s to %s. Scoring takes a few minutes.",
        args.submission,
        args.space,
    )
    reply = submit(
        args.submission,
        token=token,
        modality=args.modality,
        model_name=args.model_name,
        email=args.email,
        institution=args.institution,
        authors=args.authors,
        paper_link=args.paper_link,
        hf_model_link=args.hf_model,
        notes=args.notes,
        space=args.space,
    )
    logger.info("%s", reply["message"])
    if reply["status"] == REFUSED:
        raise SubmissionError("the Space refused the submission")
    if reply["status"] != PUBLISHED:
        raise EvaluatorError("the Space could not publish the submission")
    logger.info("Submission id: %s", reply["submission_id"])


def _add_data_dir(command: argparse.ArgumentParser, verb: str) -> None:
    """The ``--data-dir`` option, shared by every subcommand."""
    command.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"where to {verb} the data (default: {DEFAULT_DATA_DIR})",
    )


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
            "Download the datasets of one modality from the Hub, at the commit "
            "that the leaderboard scores, with the labels of its public tasks."
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
        "--revision",
        default=DATA_REVISION,
        help=(
            "the Hub commit to download (default: the one the leaderboard "
            "scores); maintainers use main for the newest data"
        ),
    )
    _add_data_dir(fetch, "put")
    fetch.set_defaults(run=_download)

    score = commands.add_parser(
        "evaluate",
        help="check a submission file and score its public tasks",
        description=(
            "Check every dataset of a submission file against the downloaded data, "
            "then score the public tasks it covers with the leaderboard protocol."
        ),
    )
    score.add_argument("submission", help="a .parquet, .csv, .tsv or .npz file")
    score.add_argument("--modality", required=True, choices=sorted(MODALITIES))
    score.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default=EMBEDDING,
        help=(
            "embedding (default, as for every submission) puts each row on the "
            "unit sphere; gene_space keeps the rows as they are, as for the "
            "HVG-1200-genes baseline"
        ),
    )
    _add_data_dir(score, "read")
    score.add_argument("-o", "--output", help="write the results to this JSON file")
    score.set_defaults(run=_evaluate)

    send = commands.add_parser(
        "submit",
        help="send a submission file to the leaderboard",
        description=(
            "Check a submission file, then send it to the leaderboard Space, which "
            "scores every task and publishes the scores. It needs a Hugging Face "
            "login: run `hf auth login` once."
        ),
    )
    send.add_argument("submission", help="a .parquet, .csv, .tsv or .npz file")
    send.add_argument("--modality", required=True, choices=sorted(MODALITIES))
    send.add_argument(
        "--model-name", required=True, help="the name shown on the leaderboard"
    )
    send.add_argument(
        "--email", required=True, help="a contact for this submission, kept private"
    )
    send.add_argument("--institution", default="", help="shown on the leaderboard")
    send.add_argument(
        "--authors", action="store_true", help="the model's authors submit it"
    )
    send.add_argument(
        "--paper-link", default="", help="links the model name on the leaderboard"
    )
    send.add_argument(
        "--hf-model", default="", help="the model's page on the Hugging Face Hub"
    )
    send.add_argument(
        "--notes", default="", help="about the model or its training data"
    )
    send.add_argument(
        "--dry-run", action="store_true", help="run every check, send nothing"
    )
    send.add_argument(
        "--space", default=SPACE, help=f"the leaderboard Space (default: {SPACE})"
    )
    _add_data_dir(send, "read")
    send.set_defaults(run=_submit)
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
