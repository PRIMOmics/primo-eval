"""Unit tests for the results schema (no network, no Gradio)."""

from pathlib import Path

import huggingface_hub
import numpy as np
import pandas as pd
import pytest
from results import (
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    IS_BASELINE,
    OWNER,
    PAPER_LINK,
    RESULT_COLUMNS,
    SUBMISSION_COLUMNS,
    SUBMISSION_ID,
    _embedding_manifest,
    display_name,
    fetch_hvg_reference,
    owner_of,
    with_author_submission_flag,
    with_baseline_flag,
    with_institution,
    with_owner,
    with_paper_link,
    with_submission_id,
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
    assert len(manifest["sha256"]) == 64
    assert b"sample_id" not in payload


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
