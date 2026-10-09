import json
import math

import numpy as np
import pandas as pd
import pytest

from primomics.boards import build_boards
from primomics.evaluate import TASK_ID
from primomics.ranking import (
    ELO_ANCHOR_MODEL,
    IS_BASELINE,
    MODEL_NAME,
    SCORE,
    SUBMITTED_AT,
    _paired_splits,
    latest_only,
    rank,
    stored_repeat_scores,
    with_baseline_flag,
)

TASKS = [
    {
        "task_id": "t901",
        "dataset_id": "d901",
        "category": "treatment_outcome",
        "metric": "auroc",
        "modality": "bulk RNA",
        "therapeutic_area": "Gastroenterology",
    },
    {
        "task_id": "t902",
        "dataset_id": "d902",
        "category": "clinical_scores",
        "metric": "pearson",
        "modality": "bulk RNA",
        "therapeutic_area": "Gastroenterology",
    },
    {
        "task_id": "t903",
        "dataset_id": "d902",
        "category": "clinical_scores",
        "metric": "pearson",
        "modality": "bulk RNA",
        "therapeutic_area": "Gastroenterology",
    },
    {
        "task_id": "t904",
        "dataset_id": "d903",
        "category": "clinical_scores",
        "metric": "pearson",
        "modality": "bulk RNA",
        "therapeutic_area": "Dermatology",
    },
    {
        "task_id": "t905",
        "dataset_id": "d904",
        "category": "endotype",
        "metric": "auroc",
        "modality": "bulk RNA",
        "therapeutic_area": "Rheumatology",
    },
]
ALL_SCORES = (0.8, 0.5, 0.6, 0.4, 0.7)


def _full(model, at, scores=ALL_SCORES, baseline=False):
    """One submission scoring every task."""
    return [
        (model, task["task_id"], score, at, baseline)
        for task, score in zip(TASKS, scores)
    ]


def _frame(*rows):
    return pd.DataFrame(
        rows, columns=[MODEL_NAME, TASK_ID, SCORE, SUBMITTED_AT, IS_BASELINE]
    )


def _on(name, tasks=TASKS):
    """The tasks of the board called ``name``."""
    board = next(board for board in build_boards(tasks) if board.name == name)
    return [task for task in tasks if task["task_id"] in board.task_ids]


def _names(df, board="bulk RNA"):
    return [entry.model_name for entry in rank(df, _on(board))]


def _elo(df, board="bulk RNA"):
    return {
        (entry.model_name, entry.is_baseline): entry.elo
        for entry in rank(df, _on(board))
    }


def test_a_model_is_ranked_only_with_every_task_of_the_board():
    df = _frame(
        *_full("full", "2026-01-01"), ("partial", "t901", 0.99, "2026-01-01", False)
    )

    assert _names(df) == ["full"]


def test_a_task_nobody_scored_does_not_block_the_ranking():
    nan = float("nan")
    df = _frame(
        *_full("better", "2026-01-01", scores=(0.9, 0.9, 0.9, 0.9, nan)),
        *_full("worse", "2026-01-01", scores=(0.2, 0.2, 0.2, 0.2, nan)),
    )

    assert _names(df) == ["better", "worse"]


def test_a_partial_model_is_ranked_on_the_board_it_covered():
    df = _frame(
        *_full("full", "2026-01-01"), ("area-only", "t905", 0.95, "2026-01-01", False)
    )

    assert _names(df) == ["full"]
    assert _names(df, board="Rheumatology") == ["area-only", "full"]


def test_the_latest_result_of_each_task_wins():
    df = _frame(
        *_full("m", "2026-01-01", scores=(0.9,) * 5),
        *_full("m", "2026-02-01", scores=(0.1,) * 5),
    )

    assert set(latest_only(df)[SUBMITTED_AT]) == {"2026-02-01"}
    assert rank(df, TASKS)[0].task_scores == {task["task_id"]: 0.1 for task in TASKS}


def test_results_sent_at_the_same_time_are_ordered_by_submission_id():
    newer = _frame(*_full("m", "2026-02-01", scores=(0.1,) * 5))
    older = _frame(*_full("m", "2026-02-01", scores=(0.9,) * 5))
    df = pd.concat(
        [
            newer.assign(submission_id="2026-02-01T00:00:00.000002-b"),
            older.assign(submission_id="2026-02-01T00:00:00.000001-a"),
        ]
    )

    assert set(rank(df, TASKS)[0].task_scores.values()) == {0.1}


def test_ids_read_back_with_spaces_still_match():
    first = _frame(*_full("m", "2026-02-01", scores=(0.9,) * 5))
    second = _frame(("m", " t901 ", 0.1, "2026-02-01", False))
    df = pd.concat(
        [second.assign(submission_id=" b "), first.assign(submission_id="a")]
    )

    assert rank(df, TASKS)[0].task_scores["t901"] == 0.1
    assert len(latest_only(df)) == len(TASKS)


def test_coverage_accumulates_across_submissions():
    rows = _full("m", "2026-01-01")
    first = _frame(*rows[:3]).assign(submission_id="first")
    second = _frame(*rows[3:]).assign(submission_id="second", submitted_at="2026-02-01")

    assert _names(pd.concat([first, second])) == ["m"]


def test_correcting_one_task_leaves_the_other_tasks_standing():
    df = _frame(*_full("m", "2026-01-01"), ("m", "t901", 0.11, "2026-02-01", False))

    scores = rank(df, TASKS)[0].task_scores

    assert scores["t901"] == 0.11
    assert scores["t905"] == ALL_SCORES[4]
    assert len(latest_only(df)) == len(TASKS)


def test_a_model_with_no_finite_score_is_not_ranked():
    df = _frame(
        *_full("nan-everywhere", "2026-01-01", scores=(float("nan"),) * 5),
        *_full("bad", "2026-01-01", scores=(-0.2,) * 5),
    )

    assert _names(df) == ["bad"]


def test_nothing_is_ranked_without_a_finite_score():
    df = _frame(*_full("nan-everywhere", "2026-01-01", scores=(float("nan"),) * 5))

    assert rank(df, TASKS) == []


def test_nothing_is_ranked_when_nobody_covered_the_board():
    df = _frame(("partial", "t901", 0.9, "2026-01-01", False))

    assert rank(df, _on("Rheumatology")) == []
    assert rank(_frame(), TASKS) == []


def test_without_the_anchor_the_tasks_somebody_scored_count():
    df = _frame(("partial", "t901", 0.99, "2026-01-01", False))

    assert _names(df) == ["partial"]


def test_a_baseline_is_ranked_in_place():
    df = _frame(
        *_full("pca-50", "2026-01-01", scores=(0.9,) * 5, baseline=True),
        *_full("model", "2026-01-01", scores=(0.4,) * 5),
    )

    entries = rank(df, TASKS)

    assert [(entry.model_name, entry.is_baseline) for entry in entries] == [
        ("pca-50", True),
        ("model", False),
    ]


def test_rows_written_without_the_flag_are_not_baselines():
    df = _frame(*_full("model", "2026-01-01")).drop(columns=IS_BASELINE)

    assert rank(df, TASKS)[0].is_baseline is False


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (True, True),
        ("True", True),
        (" true ", True),
        ("1", True),
        (1, True),
        (False, False),
        ("False", False),
        ("0", False),
        ("", False),
        (None, False),
        (float("nan"), False),
    ],
)
def test_the_baseline_flag_survives_a_csv_round_trip(stored, expected):
    df = pd.DataFrame({IS_BASELINE: [stored]}, dtype=object)

    assert with_baseline_flag(df)[IS_BASELINE].tolist() == [expected]


def test_a_baseline_and_a_model_can_share_a_name():
    df = _frame(
        *_full("pca-50", "2026-01-01", scores=(0.9,) * 5, baseline=True),
        *_full("pca-50", "2026-06-01", scores=(0.2,) * 5),
    )

    entries = rank(df, TASKS)

    assert [(entry.model_name, entry.is_baseline) for entry in entries] == [
        ("pca-50", True),
        ("pca-50", False),
    ]


def test_a_resubmission_replaces_the_entry_of_the_same_kind():
    df = _frame(
        *_full("model", "2026-01-01", scores=(0.9,) * 5),
        *_full("model", "2026-06-01", scores=(0.2,) * 5),
    )

    entries = rank(df, TASKS)

    assert len(entries) == 1
    assert set(entries[0].task_scores.values()) == {0.2}


def _anchor(at, scores=ALL_SCORES, baseline=True):
    return _full(ELO_ANCHOR_MODEL, at, scores=scores, baseline=baseline)


def test_the_anchor_holds_exactly_1000_elo():
    df = _frame(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_full("above", "2026-01-01", scores=(0.9,) * 5),
        *_full("below", "2026-01-01", scores=(0.1,) * 5),
    )

    elo = _elo(df)

    assert elo[(ELO_ANCHOR_MODEL, True)] == 1000.0
    assert elo[("above", False)] > 1000
    assert elo[("below", False)] < 1000


def test_a_weak_newcomer_does_not_move_the_anchor():
    before = _frame(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_full("model", "2026-01-01", scores=(0.9,) * 5),
    )
    after = pd.concat(
        [before, _frame(*_full("weak", "2026-06-01", scores=(0.05,) * 5))]
    )

    assert _elo(before)[(ELO_ANCHOR_MODEL, True)] == 1000.0
    assert _elo(after)[(ELO_ANCHOR_MODEL, True)] == 1000.0
    assert _elo(after)[("model", False)] > 1000


def test_a_task_the_anchor_skipped_is_not_ranked():
    df = _frame(
        *[row for row in _anchor("2026-01-01") if row[1] != "t905"],
        *[row for row in _full("partial", "2026-01-01") if row[1] != "t905"],
        *_full("full", "2026-01-01"),
    )

    entries = rank(df, TASKS)

    assert {entry.model_name for entry in entries} == {
        ELO_ANCHOR_MODEL,
        "partial",
        "full",
    }
    assert all("t905" not in entry.task_scores for entry in entries)
    assert all("endotype" not in entry.categories for entry in entries)


def test_a_submission_cannot_take_over_the_anchor():
    df = _frame(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_anchor("2026-06-01", scores=(0.9,) * 5, baseline=False),
    )

    elo = _elo(df)

    assert elo[(ELO_ANCHOR_MODEL, True)] == 1000.0
    assert elo[(ELO_ANCHOR_MODEL, False)] > 1000


def test_a_board_the_anchor_never_scored_is_centred_on_1000():
    df = _frame(
        *[row for row in _anchor("2026-01-01") if row[1] != "t905"],
        *_full("model", "2026-01-01", scores=(0.9,) * 5),
        *_full("other", "2026-01-01", scores=(0.3,) * 5),
    )

    entries = rank(df, _on("Rheumatology"))

    assert [entry.model_name for entry in entries] == ["model", "other"]
    assert np.mean([entry.elo for entry in entries]) == pytest.approx(1000.0)
    assert _elo(df)[(ELO_ANCHOR_MODEL, True)] == 1000.0


def test_without_the_anchor_the_ratings_centre_on_1000():
    df = _frame(
        *_full("a", "2026-01-01", scores=(0.9,) * 5),
        *_full("b", "2026-01-01", scores=(0.5,) * 5),
        *_full("c", "2026-01-01", scores=(0.1,) * 5),
    )

    assert np.mean(list(_elo(df).values())) == pytest.approx(1000.0)


def test_one_name_can_hold_results_on_two_modalities():
    tasks = [
        *TASKS,
        {**TASKS[0], "task_id": "t906", "dataset_id": "d905", "modality": "sc RNA"},
    ]
    df = _frame(*_full("m", "2026-01-01"), ("m", "t906", 0.42, "2026-02-01", False))

    assert [entry.model_name for entry in rank(df, _on("bulk RNA", tasks))] == ["m"]
    single_cell = rank(df, _on("sc RNA", tasks))
    assert single_cell[0].task_scores == {"t906": 0.42}


def test_ties_on_elo_go_to_the_higher_mean_of_family_means():
    """``a`` and ``b`` each win one task: ``a`` has the better mean task score,
    ``b`` the better mean of family means."""
    df = _frame(
        *_full("a", "2026-01-01", scores=(0.6, 0.8, 0.5, 0.5, 0.5)),
        *_full("b", "2026-01-01", scores=(0.95, 0.1, 0.5, 0.5, 0.5)),
    )

    entries = rank(df, TASKS)

    assert [entry.model_name for entry in entries] == ["b", "a"]
    assert [entry.elo for entry in entries] == [1000.0, 1000.0]
    assert [entry.mean_rank for entry in entries] == [1.5, 1.5]
    task_means = [np.mean(list(entry.task_scores.values())) for entry in entries]
    assert task_means == pytest.approx([0.51, 0.58])


def test_a_rival_without_splits_does_not_shrink_the_board_to_one_split():
    """``challenger`` loses split 0 to the anchor and wins the two others."""
    df = _frame(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_full("challenger", "2026-01-01", scores=(0.5667,) * 5),
    ).assign(
        repeat_scores=lambda frame: frame[MODEL_NAME].map(
            {"challenger": "[0.4, 0.6, 0.7]"}
        )
    )

    assert _elo(df)[("challenger", False)] > 1000


def test_the_shorter_record_plays_its_mean_on_every_split():
    left, right = _paired_splits((0.4, 0.6, 0.7), (0.5,))

    assert left == (0.4, 0.6, 0.7)
    assert right == (0.5, 0.5, 0.5)


def test_the_splits_of_one_task_share_one_vote():
    """``even`` wins six splits of one task and loses four tasks: it is below."""
    df = _frame(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_full("even", "2026-01-01", scores=(0.9, 0.1, 0.1, 0.1, 0.1)),
    ).assign(
        repeat_scores=lambda frame: [
            "[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]"
            if row.model_name == "even" and row.task_id == "t901"
            else ""
            for row in frame.itertuples()
        ]
    )

    assert _elo(df)[("even", False)] < 1000


@pytest.mark.parametrize(
    "stored",
    [
        "",
        float("nan"),
        "not json",
        "[]",
        '{"a": 1}',
        '["0.5"]',
        "[0.5, true]",
        "[0.5, NaN]",
        "[1e400]",
        f"[{10**400}]",
        "[" * 100_000 + "]" * 100_000,
    ],
)
def test_unreadable_split_scores_fall_back_to_the_task_score(stored):
    row = pd.Series({SCORE: 0.5, "repeat_scores": stored})

    assert stored_repeat_scores(row) == ()


def test_split_scores_are_read_back_from_json():
    row = pd.Series({SCORE: 0.5, "repeat_scores": "[0.4, 1, 0.6]"})

    assert stored_repeat_scores(row) == (0.4, 1.0, 0.6)


def _fixed_table():
    """Four models, two baselines, with 3, 10 or no split scores per task."""
    models = [
        ("HVG-1200-genes", True, (0.70, 0.40, 0.45, 0.30, 0.66), 3),
        ("model-a", False, (0.74, 0.35, 0.52, 0.41, 0.61), 3),
        ("model-b", False, (0.69, 0.47, 0.40, 0.28, 0.72), 10),
        ("pca-50", True, (0.62, 0.30, 0.38, 0.33, 0.58), 0),
    ]
    rows = []
    for model, baseline, scores, splits in models:
        for index, (task, score) in enumerate(zip(TASKS, scores)):
            repeats = [
                round(score + 0.05 * math.sin(3 * split + index), 4)
                for split in range(splits)
            ]
            rows.append(
                {
                    MODEL_NAME: model,
                    TASK_ID: task["task_id"],
                    SCORE: round(float(np.mean(repeats)), 4) if repeats else score,
                    "repeat_scores": json.dumps(repeats) if repeats else "",
                    SUBMITTED_AT: "2026-01-01",
                    IS_BASELINE: baseline,
                }
            )
    return pd.DataFrame(rows)


def test_matches_the_live_leaderboard_on_a_fixed_table():
    """Ratings the live leaderboard gives this table: a changed rule moves them."""
    entries = rank(_fixed_table(), TASKS)

    assert [
        (entry.model_name, entry.is_baseline, entry.mean_rank) for entry in entries
    ] == [
        ("model-a", False, 1.8),
        ("model-b", False, 2.4),
        ("HVG-1200-genes", True, 2.2),
        ("pca-50", True, 3.6),
    ]
    assert [entry.elo for entry in entries] == pytest.approx(
        [1089.9236126065343, 1002.6627942864257, 1000.0, 783.9415477480256],
        abs=1e-3,
    )
