"""Persisted leaderboard rows: the results-CSV schema and its Hugging Face IO.

Kept out of ``app.py`` so the schema has one owner and so the baseline publisher
can append rows without importing Gradio. ``huggingface_hub`` is imported lazily,
so the unit tests touch no network.

``is_baseline`` marks a reference submission we produced ourselves (log-normalized
expression, whole or cut to its most variable genes) rather than a model somebody
sent us. Baselines are ranked in place, never pinned: the point of showing them is
that a foundation model can lose to an information-rich expression baseline, and
a row pushed to the bottom of the table would hide exactly that.

``hf_username`` privately records who submitted a name, because the board keeps
each name's LATEST rows: without an owner, anyone could supersede another team's
ranked entry by sending a one-task file under their model name. The Submit form
refuses a name somebody else already claimed. New submissions separately store
an optional public ``institution``, an author-submission flag and a paper link;
legacy owners remain private. Historical ``team`` values are read as institutions.

Rows written before these columns existed carry no column at all, so every
reader normalises the baseline flag, ownership, public credits and submission
identity. Missing legacy public-credit values remain private.

Each row records the board ``version`` whose harness scored it, which is always
the current version of this code. A board shows only the rows of its version.
Rows written before versions existed carry none and are shown on no board.

Every submission is also scored concatenated with our ``HVG-1200-genes``
baseline features and stored as a second entry, ``<model> + HVG``, sharing the
parent's owner, credits and submission id. It shows what a model adds on top of
plain expression. The features live privately at ``HVG_REFERENCE_FILE``, next
to the results. They are read at the commit the board version pins, so a new
upload is used only once a release pins it.
"""

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from evaluator import EMBEDDING, RESULTS_REPO, load_submission
from versions import CURRENT

RESULTS_FILE = "task_results.csv"
SUBMISSIONS_FILE = "submissions.csv"

MODEL_NAME = "model_name"
IS_BASELINE = "is_baseline"
OWNER = "hf_username"
INSTITUTION = "institution"
LEGACY_TEAM = "team"
IS_AUTHOR_SUBMISSION = "is_author_submission"
PAPER_LINK = "paper_link"
SUBMISSION_ID = "submission_id"
VERSION = "version"

RESULT_COLUMNS = [
    MODEL_NAME,
    "task_id",
    "score",
    "repeat_scores",
    "diagnostics",
    "submitted_at",
    IS_BASELINE,
    OWNER,
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    PAPER_LINK,
    SUBMISSION_ID,
    VERSION,
]
SUBMISSION_COLUMNS = [
    MODEL_NAME,
    "submitted_at",
    OWNER,
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    SUBMISSION_ID,
    "email",
    PAPER_LINK,
    "hf_model_link",
    "notes",
]

TRUTHY = ("true", "1")
BASELINE_TAG = "(baseline)"
HVG_TAG = "+ HVG"
HVG_REFERENCE_MODEL = "HVG-1200-genes"
HVG_REFERENCE_FILE = f"baselines/{HVG_REFERENCE_MODEL}/submission.parquet"
ARCHIVE_DIR = "models"
ARCHIVE_MANIFEST = "manifest.json"


def with_baseline_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a boolean ``is_baseline`` column, whatever the CSV held.

    Read back from CSV the column can be bool, the strings ``True``/``False``, or
    absent on rows written before baselines existed; all of those must collapse
    to a real boolean before anything ranks on it.
    """
    if IS_BASELINE not in df.columns:
        return df.assign(**{IS_BASELINE: False})
    flags = df[IS_BASELINE].astype(str).str.strip().str.lower().isin(TRUTHY)
    return df.assign(**{IS_BASELINE: flags})


def with_owner(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a string ``hf_username`` column; an absent one means unclaimed."""
    return _with_text_column(df, OWNER)


def with_institution(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee an institution, falling back to the former ``team`` field."""
    normalized = _with_text_column(df, INSTITUTION)
    if LEGACY_TEAM not in normalized.columns:
        return normalized
    legacy = _with_text_column(normalized, LEGACY_TEAM)[LEGACY_TEAM]
    values = normalized[INSTITUTION].mask(normalized[INSTITUTION] == "", legacy)
    return normalized.assign(**{INSTITUTION: values})


def with_submission_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a submission id; legacy rows fall back to their timestamp."""
    return _with_text_column(df, SUBMISSION_ID)


def with_version(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a board-version column; legacy rows have the empty version."""
    return _with_text_column(df, VERSION)


def version_rows(df: pd.DataFrame, number: str) -> pd.DataFrame:
    """The rows that board version ``number`` scored."""
    versioned = with_version(df)
    return versioned[versioned[VERSION] == number]


def with_paper_link(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a public paper-link column; legacy rows have no public link."""
    return _with_text_column(df, PAPER_LINK)


def with_author_submission_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a boolean author-submission flag; legacy rows are unverified."""
    if IS_AUTHOR_SUBMISSION not in df.columns:
        return df.assign(**{IS_AUTHOR_SUBMISSION: False})
    flags = df[IS_AUTHOR_SUBMISSION].astype(str).str.strip().str.lower().isin(TRUTHY)
    return df.assign(**{IS_AUTHOR_SUBMISSION: flags})


def _with_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize a nullable CSV text column without stringifying missing values."""
    if column not in df.columns:
        return df.assign(**{column: ""})
    values = df[column].map(lambda value: "" if pd.isna(value) else str(value).strip())
    return df.assign(**{column: values})


def owner_of(df: pd.DataFrame, model: str) -> str:
    """Who claimed this submitted model name, or ``""`` if it is free.

    Baselines are ignored: they live in their own namespace (``is_baseline``), so
    publishing a baseline never reserves its name for user submissions.
    """
    if df.empty:
        return ""
    claimed = with_owner(with_baseline_flag(df))
    rows = claimed[
        (claimed[MODEL_NAME] == model) & ~claimed[IS_BASELINE] & (claimed[OWNER] != "")
    ]
    return "" if rows.empty else str(rows.iloc[-1][OWNER])


def display_name(model: str, is_baseline: bool) -> str:
    """Leaderboard label: a baseline says so, in the one column everybody reads."""
    return f"{model} {BASELINE_TAG}" if is_baseline else model


def hvg_name(model: str) -> str:
    """Name of the entry scoring ``model`` concatenated with the HVG features."""
    return f"{model} {HVG_TAG}"


def split_hvg_name(model: str) -> tuple[str, bool]:
    """Split ``<model> + HVG`` into the parent name and whether it is one."""
    suffix = f" {HVG_TAG}"
    if model.endswith(suffix):
        return model.removesuffix(suffix), True
    return model, False


def without_hvg(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows of every ``<model> + HVG`` entry."""
    return df[~df[MODEL_NAME].astype(str).str.endswith(f" {HVG_TAG}")]


def fetch_hvg_reference(token: str | None, revision: str = CURRENT.reference) -> Path:
    """Local path to the HVG-1200 features every submission is concatenated with."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            RESULTS_REPO,
            HVG_REFERENCE_FILE,
            repo_type="dataset",
            token=token,
            revision=revision,
        )
    )


def upload_hvg_reference(path: str | Path, token: str | None) -> str:
    """Replace the HVG-1200 features; return the commit that holds them, to pin."""
    from huggingface_hub import HfApi

    return (
        HfApi(token=token)
        .upload_file(
            path_or_fileobj=str(path),
            path_in_repo=HVG_REFERENCE_FILE,
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            commit_message="Update HVG-1200 reference features",
        )
        .oid
    )


def task_rows(
    model: str, result: dict, submitted_at: str, submission_id: str, credits: dict
) -> list[dict]:
    """One row per task scored in ``result``, credited and tagged with this version."""
    return [
        {
            MODEL_NAME: model,
            "task_id": task.task_id,
            "score": round(float(task.score), 4),
            "repeat_scores": json.dumps(
                [round(float(value), 4) for value in task.repeat_scores]
            ),
            "diagnostics": json.dumps(
                {key: round(float(value), 4) for key, value in task.diagnostics.items()}
            ),
            "submitted_at": submitted_at,
            IS_BASELINE: False,
            SUBMISSION_ID: submission_id,
            VERSION: CURRENT.number,
            **credits,
        }
        for task in result["per_task"]
    ]


def read_csv(filename: str, columns: list[str], token: str | None) -> pd.DataFrame:
    """One CSV from the private results dataset; an absent file is an empty table."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(RESULTS_REPO, filename, repo_type="dataset", token=token)
    except (RepositoryNotFoundError, EntryNotFoundError):
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, dtype={VERSION: str})


def upload_csv(filename: str, df: pd.DataFrame, token: str | None) -> None:
    """Overwrite one CSV in the private results dataset."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    api.upload_file(
        path_or_fileobj=buffer,
        path_in_repo=filename,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
    )


def archive_paths(submission_id: str, source_format: str) -> tuple[str, str]:
    """Where a submission's file and its manifest are archived in the results repo."""
    prefix = f"{ARCHIVE_DIR}/{submission_id}"
    return (
        f"{prefix}/submission{source_format or '.bin'}",
        f"{prefix}/{ARCHIVE_MANIFEST}",
    )


def _embedding_manifest(
    path: str | Path,
    model_name: str,
    submission_id: str,
    modality: str,
    submitted_at: str,
    representation: str = EMBEDDING,
) -> tuple[dict, bytes]:
    """Build a privacy-preserving manifest for an embedding submission."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    blocks = load_submission(source)
    frame = pd.concat(
        [block.assign(dataset_id=dataset_id) for dataset_id, block in blocks.items()],
        ignore_index=True,
    )
    counts = frame["dataset_id"].value_counts().sort_index().to_dict()
    embedding_columns = [
        column for column in frame.columns if column not in {"dataset_id", "sample_id"}
    ]
    manifest = {
        "model_name": model_name,
        "submission_id": submission_id,
        "submitted_at": submitted_at,
        "modality": modality,
        "representation": representation,
        "source_format": source.suffix.lower(),
        "sha256": digest.hexdigest(),
        "size_bytes": source.stat().st_size,
        "n_rows": len(frame),
        "embedding_dimension": len(embedding_columns),
        "dataset_row_counts": {str(key): int(value) for key, value in counts.items()},
    }
    return manifest, json.dumps(manifest, indent=2).encode()


def archive_submission(
    path: str | Path,
    model_name: str,
    submission_id: str,
    modality: str,
    submitted_at: str,
    token: str | None,
    representation: str = EMBEDDING,
) -> None:
    """Archive an embedding file and its manifest in the private results dataset."""
    from huggingface_hub import CommitOperationAdd, HfApi

    source = Path(path)
    _, manifest_bytes = _embedding_manifest(
        source, model_name, submission_id, modality, submitted_at, representation
    )
    file_path, manifest_path = archive_paths(submission_id, source.suffix.lower())
    HfApi(token=token).create_commit(
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=file_path, path_or_fileobj=source),
            CommitOperationAdd(
                path_in_repo=manifest_path, path_or_fileobj=manifest_bytes
            ),
        ],
        commit_message=f"Archive embedding submission {submission_id}",
    )


def read_results(token: str | None) -> pd.DataFrame:
    """Every persisted task score, with baseline, credit and version normalised."""
    return with_version(
        with_submission_id(
            with_paper_link(
                with_author_submission_flag(
                    with_institution(
                        with_owner(
                            with_baseline_flag(
                                read_csv(RESULTS_FILE, RESULT_COLUMNS, token)
                            )
                        )
                    )
                )
            )
        )
    )


def append_results(rows: list[dict], token: str | None) -> None:
    """Append scored rows to the leaderboard, keeping the history intact."""
    if not rows:
        return
    df = pd.concat([read_results(token), pd.DataFrame(rows)], ignore_index=True)
    upload_csv(
        RESULTS_FILE,
        with_version(
            with_submission_id(
                with_paper_link(
                    with_author_submission_flag(
                        with_institution(with_owner(with_baseline_flag(df)))
                    )
                )
            )
        ),
        token,
    )


def append_submission(meta: dict, token: str | None) -> None:
    """Persist private contact metadata with matching public-credit fields."""
    df = pd.concat(
        [read_csv(SUBMISSIONS_FILE, SUBMISSION_COLUMNS, token), pd.DataFrame([meta])],
        ignore_index=True,
    )
    upload_csv(SUBMISSIONS_FILE, df, token)
