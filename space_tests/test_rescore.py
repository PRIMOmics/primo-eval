"""Unit tests for the rescore tool (no network)."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
import rescore
from evaluator import (
    EMBEDDING,
    GENE_SPACE,
    EvaluatorError,
    SubmissionError,
    load_submission,
)
from rescore import Carry, carried, lost_tasks, representation_of, without_retired
from results import (
    HVG_REFERENCE_MODEL,
    IS_BASELINE,
    MODEL_NAME,
    OWNER,
    SUBMISSION_ID,
    VERSION,
    hvg_name,
)
from versions import CURRENT

TASKS = {"t901", "t902", "t903"}
DAY_1 = "2026-01-01 00:00:00.000000"
DAY_2 = "2026-01-02 00:00:00.000000"


def _rows(*rows) -> pd.DataFrame:
    """Result rows from ``(model, task_id, submitted_at, submission_id)``."""
    return pd.DataFrame(
        [
            {
                MODEL_NAME: model,
                "task_id": task_id,
                "score": 0.5,
                "submitted_at": submitted_at,
                IS_BASELINE: model == HVG_REFERENCE_MODEL,
                OWNER: "" if model == HVG_REFERENCE_MODEL else "alice",
                SUBMISSION_ID: submission_id,
            }
            for model, task_id, submitted_at, submission_id in rows
        ],
        columns=[
            MODEL_NAME,
            "task_id",
            "score",
            "submitted_at",
            IS_BASELINE,
            OWNER,
            SUBMISSION_ID,
        ],
    )


def _manifest(model: str, submission_id: str, submitted_at: str, **extra) -> dict:
    return {
        "model_name": model,
        "submission_id": submission_id,
        "submitted_at": submitted_at,
        "modality": "bulk-rna",
        "source_format": ".parquet",
        **extra,
    }


EVA = _rows(("eva", "t901", DAY_1, "s1"), ("eva", "t902", DAY_1, "s1"))
EVA_ARCHIVE = _manifest("eva", "s1", DAY_1)


def test_a_live_submission_is_carried_with_its_credits_tasks_and_date():
    carries, orphans = carried(EVA, _rows(), [EVA_ARCHIVE], TASKS, set())

    assert orphans == []
    assert carries == [
        Carry(
            model_name="eva",
            is_baseline=False,
            submission_id="s1",
            submitted_at=DAY_1,
            modality="bulk-rna",
            representation=EMBEDDING,
            archive="models/s1/submission.parquet",
            task_ids=frozenset({"t901", "t902"}),
        )
    ]
    assert carries[0].credits[OWNER] == "alice"


def test_the_tasks_a_version_adds_join_the_task_set():
    carries, _ = carried(EVA, _rows(), [EVA_ARCHIVE], TASKS, {"t903"})

    assert carries[0].task_ids == {"t901", "t902", "t903"}


def test_all_tasks_scores_every_task_of_the_file():
    carries, _ = carried(EVA, _rows(), [EVA_ARCHIVE], TASKS, set(), all_tasks=True)

    assert carries[0].task_ids is None


def test_a_retired_task_leaves_the_task_set():
    carries, _ = carried(EVA, _rows(), [EVA_ARCHIVE], {"t901"}, set())

    assert carries[0].task_ids == {"t901"}


def test_a_superseded_submission_is_not_carried():
    old = _rows(("eva", "t901", DAY_1, "s1"), ("eva", "t901", DAY_2, "s2"))
    archives = [EVA_ARCHIVE, _manifest("eva", "s2", DAY_2)]

    carries, _ = carried(old, _rows(), archives, TASKS, set())

    assert [carry.submission_id for carry in carries] == ["s2"]


def test_a_submission_the_new_version_already_holds_is_skipped():
    carries, _ = carried(EVA, EVA, [EVA_ARCHIVE], TASKS, set())

    assert carries == []


def test_a_submission_made_after_the_first_run_is_carried():
    late = _rows(("eva", "t901", DAY_2, "s2"))
    new = _rows(("eva", "t901", DAY_1, "s1"))

    carries, _ = carried(late, new, [_manifest("eva", "s2", DAY_2)], TASKS, set())

    assert [carry.submission_id for carry in carries] == ["s2"]


def test_a_plus_hvg_entry_is_not_carried_on_its_own():
    old = pd.concat([EVA, _rows((hvg_name("eva"), "t901", DAY_1, "s1"))])

    carries, _ = carried(old, _rows(), [EVA_ARCHIVE], TASKS, set())

    assert [carry.model_name for carry in carries] == ["eva"]


def test_a_live_entry_without_an_archive_is_reported():
    old = _rows(("legacy-model", "t901", DAY_1, ""))

    carries, orphans = carried(old, _rows(), [EVA_ARCHIVE], TASKS, set())

    assert carries == []
    assert orphans == ["legacy-model"]


def test_an_entry_the_new_version_holds_is_no_orphan():
    old = _rows(("legacy-model", "t901", DAY_1, ""))
    new = _rows(("legacy-model", "t901", DAY_2, "s2"))

    assert carried(old, new, [], TASKS, set()) == ([], [])


def test_the_baseline_is_carried_from_the_reference():
    old = _rows(
        (HVG_REFERENCE_MODEL, "t901", DAY_1, ""),
        (HVG_REFERENCE_MODEL, "t902", DAY_2, ""),
    )

    carries, _ = carried(old, _rows(), [], TASKS, set())

    assert carries == [
        Carry(
            model_name=HVG_REFERENCE_MODEL,
            is_baseline=True,
            submission_id="",
            submitted_at=DAY_2,
            modality=None,
            representation=GENE_SPACE,
            archive=None,
            task_ids=frozenset({"t901", "t902"}),
        )
    ]
    assert carried(old, old, [], TASKS, set()) == ([], [])


def test_an_archive_reads_its_representation_from_its_manifest():
    assert representation_of({"model_name": "eva"}) == EMBEDDING
    assert representation_of({"model_name": "COMPASS"}) == GENE_SPACE
    assert representation_of(
        {"model_name": "COMPASS", "representation": EMBEDDING}
    ) == (EMBEDDING)


def _submission(path, *dataset_ids):
    pd.DataFrame(
        {
            "dataset_id": list(dataset_ids),
            "sample_id": [f"s{i}" for i in range(len(dataset_ids))],
            "dim_0": [float(i) for i in range(len(dataset_ids))],
        }
    ).to_parquet(path)
    return path


def test_the_datasets_no_longer_listed_are_dropped(tmp_path):
    path = _submission(tmp_path / "in.parquet", "d901", "d902", "d902")

    kept, retired = without_retired(path, {"d901"}, tmp_path / "out.parquet")

    assert retired == ["d902"]
    assert list(load_submission(kept)) == ["d901"]
    assert load_submission(kept)["d901"]["dim_0"].tolist() == [0.0]


def test_a_file_with_every_dataset_listed_is_kept_as_is(tmp_path):
    path = _submission(tmp_path / "in.parquet", "d901", "d902")

    kept, retired = without_retired(path, {"d901", "d902"}, tmp_path / "out.parquet")

    assert (kept, retired) == (path, [])
    assert not (tmp_path / "out.parquet").exists()


def test_a_file_with_no_dataset_listed_is_refused(tmp_path):
    path = _submission(tmp_path / "in.parquet", "d902")

    with pytest.raises(SubmissionError, match="still listed"):
        without_retired(path, {"d901"}, tmp_path / "out.parquet")


def _fake_score_all(calls: list, invalid: frozenset[str] = frozenset()):
    def score_all(path, token, **kwargs):
        calls.append(kwargs)
        task_ids = [task["task_id"] for task in kwargs["tasks"]]
        return {
            "per_task": [
                SimpleNamespace(
                    task_id=task_id, score=0.5, repeat_scores=[], diagnostics={}
                )
                for task_id in task_ids
                if task_id not in invalid
            ],
            "invalid": [
                {"task_id": task_id, "reason": "labelled samples missing"}
                for task_id in task_ids
                if task_id in invalid
            ],
        }

    return score_all


REGISTRY = [
    {"task_id": "t901", "dataset_id": "d901"},
    {"task_id": "t902", "dataset_id": "d901"},
]


def test_a_rescored_entry_keeps_its_identity_in_the_current_version(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(rescore, "score_all", _fake_score_all(calls))
    carry = carried(EVA, _rows(), [EVA_ARCHIVE], {"t901"}, set())[0][0]
    source = _submission(tmp_path / "in.parquet", "d901")

    rows = rescore.rescore(
        carry,
        source,
        tmp_path / "hvg.parquet",
        [{"id": "d901"}],
        REGISTRY,
        tmp_path,
        None,
    )

    assert [row[MODEL_NAME] for row in rows] == ["eva", hvg_name("eva")]
    assert {row[VERSION] for row in rows} == {CURRENT.number}
    assert {row[SUBMISSION_ID] for row in rows} == {"s1"}
    assert {row["submitted_at"] for row in rows} == {DAY_1}
    assert {row[OWNER] for row in rows} == {"alice"}
    assert [call["reference"] for call in calls] == [None, tmp_path / "hvg.parquet"]
    assert [task["task_id"] for task in calls[0]["tasks"]] == ["t901"]


def test_the_rescored_baseline_stays_a_baseline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rescore, "score_all", _fake_score_all(calls))
    old = _rows((HVG_REFERENCE_MODEL, "t901", DAY_1, ""))
    carry = carried(old, _rows(), [], TASKS, set())[0][0]
    reference = _submission(tmp_path / "hvg.parquet", "d901")

    rows = rescore.rescore(
        carry, reference, reference, [{"id": "d901"}], REGISTRY, tmp_path, None
    )

    assert [row[MODEL_NAME] for row in rows] == [HVG_REFERENCE_MODEL]
    assert rows[0][IS_BASELINE] is True
    assert [(call["representation"], call["reference"]) for call in calls] == [
        (GENE_SPACE, None)
    ]


def test_a_task_scored_invalid_gets_no_row_and_is_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(rescore, "score_all", _fake_score_all([], frozenset({"t902"})))
    carry = carried(EVA, _rows(), [EVA_ARCHIVE], TASKS, set())[0][0]
    source = _submission(tmp_path / "in.parquet", "d901")

    rows = rescore.rescore(
        carry,
        source,
        tmp_path / "hvg.parquet",
        [{"id": "d901"}],
        REGISTRY,
        tmp_path,
        None,
    )

    assert {row["task_id"] for row in rows} == {"t901"}
    assert "eva: no score on t902: labelled samples missing" in caplog.text
    assert lost_tasks(carry, rows, set()) == ["t902"]


SCORED = [
    {MODEL_NAME: name, "task_id": task_id}
    for name in ("eva", hvg_name("eva"))
    for task_id in ("t901", "t902")
]


def test_a_held_task_without_a_row_is_lost_but_an_added_one_is_not():
    carry = carried(EVA, _rows(), [EVA_ARCHIVE], TASKS, {"t903"})[0][0]

    assert lost_tasks(carry, SCORED, {"t903"}) == []
    assert lost_tasks(carry, SCORED[:3], {"t903"}) == ["t902"]
    assert lost_tasks(replace(carry, task_ids=None), [], set()) == []


def _patch_main(monkeypatch, tmp_path, score) -> list:
    published = []
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(sys, "argv", ["rescore.py", "--from", "legacy", "--publish"])
    monkeypatch.setattr(rescore, "fetch_manifest", lambda token: [{"id": "d901"}])
    monkeypatch.setattr(rescore, "fetch_tasks_registry", lambda token: REGISTRY)
    monkeypatch.setattr(rescore, "read_results", lambda token: EVA)
    monkeypatch.setattr(rescore, "_manifests", lambda token: [EVA_ARCHIVE])
    monkeypatch.setattr(rescore, "fetch_hvg_reference", lambda token: tmp_path)
    monkeypatch.setattr(rescore, "_download", lambda path, token: tmp_path)
    monkeypatch.setattr(rescore, "rescore", score)
    monkeypatch.setattr(
        rescore, "append_results", lambda rows, token: published.append(rows)
    )
    return published


def test_the_rows_are_published_when_every_entry_is_scored(tmp_path, monkeypatch):
    published = _patch_main(monkeypatch, tmp_path, lambda carry, *args: SCORED)

    rescore.main()

    assert published == [SCORED]


def _fail(carry, *args):
    raise EvaluatorError("reference features lack dataset d901")


@pytest.mark.parametrize("score", [_fail, lambda carry, *args: SCORED[:3]])
def test_a_failed_entry_stops_the_publish(tmp_path, monkeypatch, score):
    published = _patch_main(monkeypatch, tmp_path, score)

    with pytest.raises(SystemExit, match="nothing is published"):
        rescore.main()

    assert published == []
