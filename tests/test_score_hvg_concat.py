import json
from types import SimpleNamespace

import pandas as pd
from score_hvg_concat import hvg_rows, live_submissions


def _row(
    model, submission_id, at, is_baseline=False, owner="lab", task_id="t901"
) -> dict:
    return {
        "model_name": model,
        "task_id": task_id,
        "score": 0.5,
        "submitted_at": at,
        "is_baseline": is_baseline,
        "hf_username": owner,
        "institution": "Lab",
        "is_author_submission": True,
        "paper_link": "https://example.org/paper",
        "submission_id": submission_id,
    }


def _manifest(model, submission_id) -> dict:
    return {
        "model_name": model,
        "submission_id": submission_id,
        "modality": "bulk-rna",
        "source_format": ".parquet",
    }


def test_only_live_plain_submissions_are_backfilled():
    results = pd.DataFrame(
        [
            _row("model-a", "old", "2026-01-01"),
            _row("model-a", "new", "2026-02-01"),
            _row("model-a + HVG", "new", "2026-02-01"),
            _row("HVG-1200-genes", "", "2026-01-01", is_baseline=True, owner=""),
        ]
    )
    manifests = [
        _manifest("model-a", "old"),
        _manifest("model-a", "new"),
        _manifest("model-a + HVG", "new"),
    ]

    live = live_submissions(results, manifests, {"t901"})

    assert [item.manifest["submission_id"] for item in live] == ["new"]
    assert live[0].manifest["model_name"] == "model-a"


def test_submissions_left_only_on_retired_tasks_are_not_backfilled():
    results = pd.DataFrame(
        [
            _row("model-a", "live", "2026-01-01", task_id="t902"),
            _row("model-b", "retired", "2026-01-01", task_id="t901"),
        ]
    )
    manifests = [_manifest("model-a", "live"), _manifest("model-b", "retired")]

    live = live_submissions(results, manifests, {"t902"})

    assert [item.manifest["submission_id"] for item in live] == ["live"]


def test_hvg_rows_are_credited_and_identified_as_their_parent():
    results = pd.DataFrame([_row("model-a", "sid", "2026-01-01")])
    [submission] = live_submissions(results, [_manifest("model-a", "sid")], {"t901"})
    result = {
        "per_task": [
            SimpleNamespace(
                task_id="t901", score=0.81234, repeat_scores=(0.8, 0.82), diagnostics={}
            )
        ]
    }

    [row] = hvg_rows(submission, result, "2026-03-01 00:00:00.000000")

    assert row["model_name"] == "model-a + HVG"
    assert row["submission_id"] == "sid"
    assert row["hf_username"] == "lab"
    assert row["institution"] == "Lab"
    assert row["paper_link"] == "https://example.org/paper"
    assert bool(row["is_author_submission"]) is True
    assert row["is_baseline"] is False
    assert row["score"] == 0.8123
    assert json.loads(row["repeat_scores"]) == [0.8, 0.82]
