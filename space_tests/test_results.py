"""Unit tests for the results schema (no network, no Gradio)."""

import io
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pandas as pd
import pytest
from evaluator import EvaluatorError
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from results import (
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    IS_BASELINE,
    OWNER,
    PAPER_LINK,
    RESULT_COLUMNS,
    RESULTS_FILE,
    SUBMISSION_COLUMNS,
    SUBMISSION_ID,
    SUBMISSIONS_FILE,
    VERSION,
    WRITE_ATTEMPTS,
    _embedding_manifest,
    append_results,
    append_submission,
    archive_paths,
    archive_submission,
    display_name,
    fetch_hvg_reference,
    owner_of,
    read_csv,
    task_rows,
    version_rows,
    with_author_submission_flag,
    with_baseline_flag,
    with_institution,
    with_owner,
    with_paper_link,
    with_submission_id,
    with_version,
    without_hvg,
)
from versions import CURRENT


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame({"model_name": ["m"], "score": [0.5], **columns})


def _rows(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["model_name", IS_BASELINE, OWNER])


def test_a_missing_column_means_nobody_is_a_baseline():
    flagged = with_baseline_flag(_frame())
    assert list(flagged[IS_BASELINE]) == [False]


def test_booleans_survive_the_round_trip():
    flagged = with_baseline_flag(_frame(**{IS_BASELINE: [True]}))
    assert list(flagged[IS_BASELINE]) == [True]


def test_the_strings_a_csv_gives_back_are_read_as_booleans():
    """``read_csv`` yields ``"True"``/``"False"`` as soon as one row is blank."""
    flagged = with_baseline_flag(
        pd.DataFrame({IS_BASELINE: ["True", "false", "TRUE ", np.nan, ""]})
    )
    assert list(flagged[IS_BASELINE]) == [True, False, True, False, False]


def test_submission_identity_is_part_of_both_schemas():
    assert {
        IS_BASELINE,
        OWNER,
        INSTITUTION,
        IS_AUTHOR_SUBMISSION,
        PAPER_LINK,
        SUBMISSION_ID,
    } <= set(RESULT_COLUMNS)
    assert {OWNER, INSTITUTION, IS_AUTHOR_SUBMISSION, PAPER_LINK, SUBMISSION_ID} <= set(
        SUBMISSION_COLUMNS
    )
    assert "diagnostics" in RESULT_COLUMNS


def test_only_baselines_are_labelled():
    assert display_name("pca-50", True) == "pca-50 (baseline)"
    assert display_name("eva-rna-v1", False) == "eva-rna-v1"


def test_rows_written_before_the_owner_existed_are_unclaimed():
    assert list(with_owner(_frame())[OWNER]) == [""]
    assert list(with_owner(_frame(**{OWNER: [np.nan]}))[OWNER]) == [""]


def test_rows_written_before_the_institution_existed_have_no_institution():
    assert list(with_institution(_frame())[INSTITUTION]) == [""]
    assert list(with_institution(_frame(**{INSTITUTION: [np.nan]}))[INSTITUTION]) == [
        ""
    ]


def test_legacy_team_values_are_read_as_institutions():
    frame = _frame(team=["Legacy Lab"])
    assert list(with_institution(frame)[INSTITUTION]) == ["Legacy Lab"]


def test_explicit_institution_wins_over_legacy_team():
    frame = _frame(institution=["Current Lab"], team=["Old Lab"])
    assert list(with_institution(frame)[INSTITUTION]) == ["Current Lab"]


def test_legacy_rows_are_not_marked_as_author_submissions():
    assert list(with_author_submission_flag(_frame())[IS_AUTHOR_SUBMISSION]) == [False]
    assert list(
        with_author_submission_flag(_frame(**{IS_AUTHOR_SUBMISSION: [np.nan]}))[
            IS_AUTHOR_SUBMISSION
        ]
    ) == [False]


def test_nullable_public_credits_and_submission_ids_normalize_to_blank():
    frame = _frame(
        **{PAPER_LINK: [pd.NA], INSTITUTION: [pd.NA], SUBMISSION_ID: [pd.NA]}
    )
    assert list(with_paper_link(frame)[PAPER_LINK]) == [""]
    assert list(with_institution(frame)[INSTITUTION]) == [""]
    assert list(with_submission_id(frame)[SUBMISSION_ID]) == [""]


def test_a_name_is_owned_by_whoever_submitted_it():
    assert owner_of(_rows(("eva", False, "scienta")), "eva") == "scienta"


def test_an_unclaimed_or_unknown_name_is_free():
    assert owner_of(_rows(("eva", False, "")), "eva") == ""
    assert owner_of(_rows(("eva", False, "scienta")), "other") == ""
    assert owner_of(pd.DataFrame(), "eva") == ""


def test_publishing_a_baseline_never_locks_the_name_for_submitters():
    """``pca-50`` the baseline and ``pca-50`` the model are separate entries."""
    assert owner_of(_rows(("pca-50", True, "scienta")), "pca-50") == ""


def test_embedding_manifest_contains_checksum_and_coverage(tmp_path):
    path = tmp_path / "submission.parquet"
    pd.DataFrame(
        {
            "dataset_id": ["d901", "d901", "d902"],
            "sample_id": ["s1", "s2", "s3"],
            "embedding_0000": [1, 2, 3],
        }
    ).to_parquet(path)
    manifest, payload = _embedding_manifest(
        path, "model", "submission", "bulk-rna", "2026-01-01"
    )
    assert manifest["n_rows"] == 3
    assert manifest["embedding_dimension"] == 1
    assert manifest["dataset_row_counts"] == {"d901": 2, "d902": 1}
    assert manifest["representation"] == "embedding"
    assert len(manifest["sha256"]) == 64
    assert b"sample_id" not in payload


def test_embedding_manifest_records_a_gene_space_representation(tmp_path):
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        {"dataset_id": ["d901"], "sample_id": ["s1"], "gene_0000": [1.0]}
    ).to_csv(path, index=False)
    manifest, _ = _embedding_manifest(
        path, "model", "submission", "bulk-rna", "2026-01-01", "gene_space"
    )
    assert manifest["representation"] == "gene_space"


@pytest.mark.parametrize("extension", [".csv", ".npz"])
def test_embedding_manifest_supports_all_submission_formats(tmp_path, extension):
    path = tmp_path / f"submission{extension}"
    if extension == ".csv":
        pd.DataFrame(
            {
                "dataset_id": ["d901", "d901"],
                "sample_id": ["s1", "s2"],
                "embedding_0000": [1, 2],
            }
        ).to_csv(path, index=False)
    else:
        np.savez(
            path,
            dataset_ids=["d901", "d901"],
            sample_ids=["s1", "s2"],
            embeddings=[[1, 2], [3, 4]],
        )
    manifest, _ = _embedding_manifest(
        path, "model", "submission", "bulk-rna", "2026-01-01"
    )
    assert manifest["n_rows"] == 2
    assert manifest["embedding_dimension"] == (1 if extension == ".csv" else 2)
    assert manifest["dataset_row_counts"] == {"d901": 2}


def test_without_hvg_drops_only_hvg_entries():
    df = pd.DataFrame(
        {"model_name": ["m", "m + HVG", "HVG-1200-genes", "m + HVG v2"], "score": 0.5}
    )
    assert list(without_hvg(df)["model_name"]) == ["m", "HVG-1200-genes", "m + HVG v2"]
    assert without_hvg(pd.DataFrame(columns=RESULT_COLUMNS)).empty


def test_the_hvg_reference_is_read_at_the_commit_of_the_current_version(monkeypatch):
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append(kwargs["revision"])
        return "hvg.parquet"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    assert fetch_hvg_reference(None) == Path("hvg.parquet")
    assert calls == [CURRENT.reference]


def test_rows_written_before_versions_existed_have_the_empty_version():
    assert list(with_version(_frame())[VERSION]) == [""]
    assert list(with_version(_frame(**{VERSION: [np.nan]}))[VERSION]) == [""]


def test_a_version_keeps_only_its_own_rows():
    df = pd.DataFrame({"model_name": ["a", "b", "c"], VERSION: ["0.1", "0.2", np.nan]})
    assert list(version_rows(df, "0.1")["model_name"]) == ["a"]
    assert list(version_rows(df, "")["model_name"]) == ["c"]


def test_scored_rows_are_tagged_with_the_current_version():
    result = {
        "per_task": [
            SimpleNamespace(task_id="t901", score=0.5, repeat_scores=[], diagnostics={})
        ]
    }
    rows = task_rows("m", result, "2026-01-01", "s", {OWNER: "alice"})
    assert [row[VERSION] for row in rows] == [CURRENT.number]


def test_versions_are_read_back_as_text(tmp_path, monkeypatch):
    path = tmp_path / "task_results.csv"
    path.write_text("model_name,version\na,0.10\nb,\n")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: path)

    df = with_version(read_csv("task_results.csv", RESULT_COLUMNS, None))

    assert list(df[VERSION]) == ["0.10", ""]


def test_a_submission_is_archived_where_the_rescore_reads_it(tmp_path, monkeypatch):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"dataset_id": ["d901"], "sample_id": ["s1"], "dim_0": [1.0]}).to_csv(
        path, index=False
    )
    commits = []

    class Api:
        def __init__(self, token):
            pass

        def create_commit(self, operations, **kwargs):
            commits.append([operation.path_in_repo for operation in operations])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)

    archive_submission(path, "m", "s1", "bulk-rna", "2026-01-01", None)

    assert commits == [list(archive_paths("s1", ".csv"))]
    assert archive_paths("s1", ".csv") == (
        "models/s1/submission.csv",
        "models/s1/manifest.json",
    )
    assert archive_paths("s1", "")[0] == "models/s1/submission.bin"


def _response(status_code: int) -> SimpleNamespace:
    return SimpleNamespace(status_code=status_code, headers={}, request=None)


class FakeResultsRepo:
    """The private results dataset: the files of each commit, oldest first."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.commits = [{}]
        self.uploads = 0

    def __call__(self, token=None):
        return self

    @property
    def head(self) -> str:
        return f"{len(self.commits) - 1:040x}"

    def create_repo(self, repo_id, **kwargs):
        pass

    def dataset_info(self, repo_id):
        return SimpleNamespace(sha=self.head)

    def upload_file(self, *, path_or_fileobj, path_in_repo, parent_commit=None, **kw):
        self.uploads += 1
        if parent_commit not in (None, self.head):
            raise HfHubHTTPError("the branch moved", response=_response(412))
        self.commits.append({**self.commits[-1], path_in_repo: path_or_fileobj.read()})

    def download(self, repo_id, filename, revision=None, **kwargs):
        revision = revision or self.head
        files = self.commits[int(revision, 16)]
        if filename not in files:
            raise EntryNotFoundError(filename)
        path = self.folder / f"{revision}-{filename}"
        path.write_bytes(files[filename])
        return str(path)

    def table(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.commits[-1][filename]))


@pytest.fixture
def results_repo(tmp_path, monkeypatch):
    """Route the Hub calls of ``results`` to a ``FakeResultsRepo``."""
    repo = FakeResultsRepo(tmp_path)
    monkeypatch.setattr(huggingface_hub, "HfApi", repo)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", repo.download)
    return repo


APPENDS = {
    RESULTS_FILE: lambda name: append_results(
        [{"model_name": name, "task_id": "t901"}], None
    ),
    SUBMISSIONS_FILE: lambda name: append_submission({"model_name": name}, None),
}


@pytest.mark.parametrize("filename", [RESULTS_FILE, SUBMISSIONS_FILE])
def test_two_appends_that_cross_keep_every_row(results_repo, monkeypatch, filename):
    """The other write lands between the read and the upload of ours."""
    results_repo.commits = [{filename: b"model_name\nolder\n"}]
    reads = []

    def read_then_let_another_write_land(*args, **kwargs):
        path = results_repo.download(*args, **kwargs)
        reads.append(path)
        if len(reads) == 1:
            APPENDS[filename]("other")
        return path

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", read_then_let_another_write_land
    )

    APPENDS[filename]("mine")

    assert list(results_repo.table(filename)["model_name"]) == [
        "older",
        "other",
        "mine",
    ]


def test_an_append_stops_when_a_write_lands_before_each_attempt(
    results_repo, monkeypatch
):
    upload = results_repo.upload_file

    def upload_after_another_write(**kwargs):
        results_repo.commits.append(results_repo.commits[-1])
        upload(**kwargs)

    monkeypatch.setattr(results_repo, "upload_file", upload_after_another_write)

    with pytest.raises(EvaluatorError, match="rows were not saved"):
        APPENDS[RESULTS_FILE]("mine")

    assert results_repo.uploads == WRITE_ATTEMPTS
    assert RESULTS_FILE not in results_repo.commits[-1]


def test_an_append_stops_at_once_on_another_hub_error(results_repo, monkeypatch):
    def refuse(**kwargs):
        results_repo.uploads += 1
        raise HfHubHTTPError("unauthorized", response=_response(401))

    monkeypatch.setattr(results_repo, "upload_file", refuse)

    with pytest.raises(HfHubHTTPError, match="unauthorized"):
        APPENDS[SUBMISSIONS_FILE]("mine")

    assert results_repo.uploads == 1
