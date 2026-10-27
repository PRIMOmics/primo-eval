"""Unit tests for the board tables and page copy (no network, no Gradio)."""

import re

import pandas as pd
import pytest
from boards import build_boards
from leaderboard import (
    ELO_ANCHOR_MODEL,
    RESERVED_COLUMNS,
    TASK_COLUMNS,
    _entries,
    _paired_splits,
    latest_only,
    per_task_table,
    ranked_table,
    source_repositories,
    submission_counts,
    tasks_table,
)

from .conftest import REGISTRY

ALL_SCORES = (0.8, 0.5, 0.6, 0.4, 0.7)


def _full(model: str, at: str, scores=ALL_SCORES) -> list[tuple]:
    return [(model, tid, s, at) for tid, s in zip(REGISTRY, scores)]


def _flagged(model: str, at: str, baseline: bool, scores=ALL_SCORES) -> list[tuple]:
    return [(model, tid, s, at, baseline) for tid, s in zip(REGISTRY, scores)]


def test_full_board_coverage_is_ranked_and_partial_is_not(registry, named, results):
    df = results(*_full("full", "2026-01-01"), ("partial", "t901", 0.99, "2026-01-01"))
    board = ranked_table(df, registry, named("bulk RNA"))
    assert list(board["Model"]) == ["full"]


def test_tasks_with_no_finite_score_do_not_block_rankings(registry, named, results):
    df = results(
        *_full("better", "2026-01-01", scores=(0.9, 0.9, 0.9, 0.9, float("nan"))),
        *_full("worse", "2026-01-01", scores=(0.2, 0.2, 0.2, 0.2, float("nan"))),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert list(board["Model"]) == ["better", "worse"]


def test_a_partial_model_is_ranked_on_the_board_it_fully_covered(
    registry, named, results
):
    """Covering all of Rheumatology ranks you there, even with Dermatology missing."""
    df = results(
        *_full("full", "2026-01-01"),
        ("rheuma_only", "t905", 0.95, "2026-01-01"),
    )
    assert list(ranked_table(df, registry, named("bulk RNA"))["Model"]) == ["full"]
    rheumatology = ranked_table(df, registry, named("Rheumatology"))
    assert list(rheumatology["Model"]) == ["rheuma_only", "full"]


def test_ranks_are_ordered_by_the_mean_of_the_category_means(registry, named, results):
    df = results(
        *_full("better", "2026-01-01", scores=(0.9, 0.9, 0.9, 0.9, 0.9)),
        *_full("worse", "2026-01-01", scores=(0.2, 0.2, 0.2, 0.2, 0.2)),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert list(board["Model"]) == ["better", "worse"]
    assert list(board["Rank"]) == [1, 2]


def test_one_column_per_category_in_its_native_metric(registry, named, results):
    df = results(*_full("full", "2026-01-01"))
    board = ranked_table(df, registry, named("bulk RNA"))
    assert "Treatment outcome (AUROC)" in board.columns
    assert "Clinical scores (Pearson)" in board.columns
    assert "Endotype (AUROC)" in board.columns


def test_comparison_columns_are_present_for_every_board(registry, named, results):
    df = results(*_full("full", "2026-01-01"))
    bulk = ranked_table(df, registry, named("bulk RNA"))
    endotype = ranked_table(df, registry, named("Endotype"))
    for table in (bulk, endotype):
        assert {"Elo", "Mean rank"} <= set(table.columns)
        assert "Mean score" not in table.columns
    assert "Endotype (AUROC)" in endotype.columns


def test_latest_submission_wins_so_scores_cannot_be_shopped(registry, named, results):
    df = results(
        *_full("m", "2026-01-01", scores=(0.9,) * 5),
        *_full("m", "2026-02-01", scores=(0.1,) * 5),
    )
    assert set(latest_only(df)["submitted_at"]) == {"2026-02-01"}
    board = ranked_table(df, registry, named("bulk RNA"))
    assert board.loc[0, "Treatment outcome (AUROC)"] == pytest.approx(0.1)


def test_equal_timestamp_submissions_do_not_mix_scores_or_credits(
    registry, named, results
):
    older = results(*_full("m", "2026-02-01", scores=(0.9,) * 5)).assign(
        submission_id="2026-02-01T00:00:00.000001-a",
        institution="Older University",
        is_author_submission=False,
        paper_link="https://example.org/older",
    )
    newer = results(*_full("m", "2026-02-01", scores=(0.1,) * 5)).assign(
        submission_id="2026-02-01T00:00:00.000002-b",
        institution="Newer University",
        is_author_submission=True,
        paper_link="https://example.org/newer",
    )
    board = ranked_table(pd.concat([older, newer]), registry, named("bulk RNA"))
    assert board.loc[0, "Treatment outcome (AUROC)"] == pytest.approx(0.1)
    assert "Author" not in board
    assert board.loc[0, "Institution"] == "Newer University"
    assert board.loc[0, "Model"].paper_link == "https://example.org/newer"
    assert board.loc[0, "Model"].is_author_submission is True


def test_one_submission_can_be_completed_at_a_later_timestamp(registry, named, results):
    rows = _full("m", "2026-01-01")
    initial = results(*rows[:-1]).assign(submission_id="one-submission")
    additional = results(rows[-1]).assign(submission_id="one-submission")
    additional["submitted_at"] = "2026-02-01"
    df = pd.concat([initial, additional], ignore_index=True)

    board = named("bulk RNA")
    assert list(ranked_table(df, registry, board)["Model"]) == ["m"]
    assert _entries(df, registry, board)[0]["submitted_at"] == "2026-02-01"


def test_nullable_credits_render_as_missing(registry, named, results):
    df = results(*_full("m", "2026-01-01")).assign(
        institution=float("nan"),
        paper_link=pd.NA,
        is_author_submission=pd.NA,
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert pd.isna(board.loc[0, "Institution"])
    assert board.loc[0, "Model"].paper_link == ""
    assert board.loc[0, "Model"].is_author_submission is False


def test_a_model_scoring_nothing_finite_is_not_ranked(registry, named, results):
    """A submission with no finite score has no scored task coverage."""
    df = results(
        *_full("nan-everywhere", "2026-01-01", scores=(float("nan"),) * 5),
        *_full("bad", "2026-01-01", scores=(-0.2,) * 5),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert list(board["Model"]) == ["bad"]


def test_ranked_is_empty_when_nobody_covered_the_board(registry, named, results):
    df = results(("partial", "t901", 0.9, "2026-01-01"))
    assert ranked_table(df, registry, named("Rheumatology")).empty


def test_per_task_keeps_partial_submissions_with_gaps(registry, named, results):
    df = results(
        *_full("full", "2026-01-01"),
        ("partial", "t902", 0.95, "2026-01-01"),
    )
    table = per_task_table(df, registry, named("Gastroenterology")).set_index("Task")
    assert list(table.index) == [
        "IBD adalimumab remission",
        "Crohn SES-CD",
        "Crohn HBI",
    ]
    assert table.loc["Crohn SES-CD", "partial"] == pytest.approx(0.95)
    assert table.loc["Crohn SES-CD", "Best"] == "partial"
    assert pd.isna(table.loc["Crohn HBI", "partial"])
    assert table.loc["Crohn HBI", "Best"] == "full"


def test_per_task_exposes_sd_only_for_multiple_stored_repeats(registry, named, results):
    df = results(
        ("repeated", "t901", 0.7, "2026-01-01"),
        ("single", "t901", 0.6, "2026-01-01"),
        ("missing", "t901", 0.5, "2026-01-01"),
        ("object", "t901", 0.5, "2026-01-01"),
        ("string", "t901", 0.5, "2026-01-01"),
        ("booleans", "t901", 0.5, "2026-01-01"),
        ("huge", "t901", 0.5, "2026-01-01"),
        ("deep", "t901", 0.5, "2026-01-01"),
    ).assign(
        repeat_scores=[
            "[0.6, 0.8]",
            "[0.6]",
            "",
            '{"0": 0.6, "1": 0.8}',
            '"12"',
            "[true, false]",
            f"[1{'0' * 400}, 2]",
            "[" * 1_000 + "0" + "]" * 1_000,
        ]
    )

    table = per_task_table(df, registry, named("Gastroenterology"))

    standard_deviation, repeats = table.attrs["repeat_standard_deviations"][
        (0, "repeated")
    ]
    assert standard_deviation == pytest.approx(0.1)
    assert repeats == 2
    assert len(table.attrs["repeat_standard_deviations"]) == 1


def test_per_task_shows_only_the_metadata_that_varies(registry, named, results):
    df = results(("m", "t904", 0.5, "2026-01-01"))
    dermatology = per_task_table(df, registry, named("Dermatology"))
    bulk = per_task_table(df, registry, named("bulk RNA"))

    assert {"Area", "Family", "Modality"}.isdisjoint(dermatology.columns)
    assert {"Area", "Family"} <= set(bulk.columns)
    assert "Modality" not in bulk.columns


def test_category_boards_keep_each_modality_to_its_own_tasks(registry, results):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "Single Cell RNA",
        "title": "Single-cell RA endotype",
    }
    boards = [board for board in build_boards(registry) if board.name == "Endotype"]
    tables = {
        board.modality: per_task_table(
            results(
                ("m", "t905", 0.5, "2026-01-01"),
                ("m", "t906", 0.6, "2026-01-01"),
            ),
            registry,
            board,
        ).set_index("Task")
        for board in boards
    }

    assert list(tables["bulk RNA"].index) == ["RA blood endotype"]
    assert list(tables["Single Cell RNA"].index) == ["Single-cell RA endotype"]


def test_reserved_names_cover_the_metadata_columns_of_every_board_type(
    registry, named, results
):
    """A model named after one of these would overwrite it -- Submit refuses them."""
    df = results(("m", "t904", 0.5, "2026-01-01"))
    area = per_task_table(df, registry, named("Dermatology"))
    modality = per_task_table(df, registry, named("bulk RNA"))
    metadata = (set(area.columns) | set(modality.columns)) - {"m"}

    assert metadata <= RESERVED_COLUMNS


def test_per_task_only_shows_the_board_it_was_asked_for(registry, named, results):
    df = results(*_full("full", "2026-01-01"))
    tasks = set(per_task_table(df, registry, named("Dermatology"))["Task"])
    assert tasks == {"Psoriasis PASI"}


def test_per_task_is_empty_when_no_result_touches_the_board(registry, named, results):
    df = results(("partial", "t901", 0.9, "2026-01-01"))
    assert per_task_table(df, registry, named("Rheumatology")).empty


def test_a_baseline_is_ranked_in_place_and_labelled(registry, named, flagged_results):
    """A PCA that beats a submitted model belongs above it, not in a footnote."""
    df = flagged_results(
        *_flagged("pca-50", "2026-01-01", True, scores=(0.9,) * 5),
        *_flagged("eva", "2026-01-01", False, scores=(0.4,) * 5),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert list(board["Model"]) == ["pca-50 (baseline)", "eva"]
    assert list(board["Rank"]) == [1, 2]


def test_rows_written_before_the_flag_existed_are_not_baselines(
    registry, named, results
):
    df = results(*_full("eva", "2026-01-01"))
    assert list(ranked_table(df, registry, named("bulk RNA"))["Model"]) == ["eva"]


def test_a_model_cannot_bury_a_baseline_by_reusing_its_name(
    registry, named, flagged_results
):
    """Same name, different kind: two entries, and the later one hides nothing."""
    df = flagged_results(
        *_flagged("pca-50", "2026-01-01", True, scores=(0.9,) * 5),
        *_flagged("pca-50", "2026-06-01", False, scores=(0.2,) * 5),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert list(board["Model"]) == ["pca-50 (baseline)", "pca-50"]
    assert "pca-50 (baseline)" in per_task_table(df, registry, named("Dermatology"))


def test_a_resubmission_still_replaces_the_entry_of_the_same_kind(
    registry, named, flagged_results
):
    df = flagged_results(
        *_flagged("eva", "2026-01-01", False, scores=(0.9,) * 5),
        *_flagged("eva", "2026-06-01", False, scores=(0.2,) * 5),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert list(board["Model"]) == ["eva"]
    assert board.loc[0, "Endotype (AUROC)"] == pytest.approx(0.2)


def test_per_task_columns_name_the_baseline(registry, named, flagged_results):
    df = flagged_results(*_flagged("random-32", "2026-01-01", True))
    table = per_task_table(df, registry, named("Dermatology"))
    assert "random-32 (baseline)" in table.columns
    assert table.loc[0, "Best"] == "random-32 (baseline)"


def _anchor(at: str, baseline: bool = True, scores=ALL_SCORES) -> list[tuple]:
    return _flagged(ELO_ANCHOR_MODEL, at, baseline, scores=scores)


def _elo(df, registry, board) -> dict[str, int]:
    table = ranked_table(df, registry, board)
    return dict(zip(table["Model"], table["Elo"]))


def test_the_anchor_holds_exactly_1000_elo(registry, named, flagged_results):
    """The fixed point of the scale: whoever else is ranked, the anchor is 1000."""
    df = flagged_results(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_flagged("above", "2026-01-01", False, scores=(0.9,) * 5),
        *_flagged("below", "2026-01-01", False, scores=(0.1,) * 5),
    )
    elo = _elo(df, registry, named("bulk RNA"))
    assert elo[f"{ELO_ANCHOR_MODEL} (baseline)"] == 1000
    assert elo["above"] > 1000
    assert elo["below"] < 1000


def test_a_weak_newcomer_does_not_lift_the_board(registry, named, flagged_results):
    """Mean-centring repriced every model on each publication; the anchor pins it."""
    board = named("bulk RNA")
    before = flagged_results(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_flagged("eva", "2026-01-01", False, scores=(0.9,) * 5),
    )
    after = flagged_results(
        *before.itertuples(index=False, name=None),
        *_flagged("weak", "2026-06-01", False, scores=(0.05,) * 5),
    )
    anchor_label = f"{ELO_ANCHOR_MODEL} (baseline)"
    assert _elo(before, registry, board)[anchor_label] == 1000
    assert _elo(after, registry, board)[anchor_label] == 1000
    assert _elo(after, registry, board)["eva"] > 1000


def test_a_task_the_anchor_skipped_is_not_ranked(registry, named, flagged_results):
    """The anchor defines the ranked task set, so t905 leaves with it."""
    without_t905 = [row for row in _anchor("2026-01-01") if row[1] != "t905"]
    df = flagged_results(
        *without_t905,
        *[row for row in _flagged("partial", "2026-01-01", False) if row[1] != "t905"],
        *_flagged("full", "2026-01-01", False),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert set(board["Model"]) == {
        f"{ELO_ANCHOR_MODEL} (baseline)",
        "partial",
        "full",
    }
    assert "Endotype (AUROC)" not in board.columns


def test_a_submission_cannot_take_over_the_anchor(registry, named, flagged_results):
    """Same name, but only the published baseline defines the scale."""
    df = flagged_results(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_anchor("2026-06-01", baseline=False, scores=(0.9,) * 5),
    )
    elo = _elo(df, registry, named("bulk RNA"))
    assert elo[f"{ELO_ANCHOR_MODEL} (baseline)"] == 1000
    assert elo[ELO_ANCHOR_MODEL] > 1000


def test_a_board_the_anchor_never_scored_still_ranks(registry, named, flagged_results):
    """A task family added since the baseline last ran must not blank its board.

    The anchor holds t901-t904, so Rheumatology and Endotype (t905 only) have no
    anchor at all. They fall back to a mean-centred scale instead of vanishing.
    """
    df = flagged_results(
        *[row for row in _anchor("2026-01-01") if row[1] != "t905"],
        *_flagged("eva", "2026-01-01", False, scores=(0.9,) * 5),
        *_flagged("other", "2026-01-01", False, scores=(0.3,) * 5),
    )
    rheumatology = ranked_table(df, registry, named("Rheumatology"))
    assert list(rheumatology["Model"]) == ["eva", "other"]
    assert rheumatology["Elo"].mean() == pytest.approx(1000, abs=1)
    assert (
        _elo(df, registry, named("bulk RNA"))[f"{ELO_ANCHOR_MODEL} (baseline)"] == 1000
    )


def test_without_the_anchor_the_ratings_centre_on_1000(registry, named, results):
    """Baseline not published yet: the board still ranks, on a mean-centred scale."""
    df = results(
        *_full("a", "2026-01-01", scores=(0.9,) * 5),
        *_full("b", "2026-01-01", scores=(0.5,) * 5),
        *_full("c", "2026-01-01", scores=(0.1,) * 5),
    )
    board = ranked_table(df, registry, named("bulk RNA"))
    assert board["Elo"].mean() == pytest.approx(1000, abs=1)


def test_tasks_table_shows_biology_but_never_provenance(registry):
    table = tasks_table(list(registry.values())).set_index("Task")
    row = table.loc["IBD adalimumab remission"]
    assert row["Disease"] == "Crohn Disease, Ulcerative Colitis"
    assert row["Tissue"] == "Digestive System"
    assert row["Patients"] == 90
    assert row["Metric"] == "AUROC"
    assert row["Family"] == "Treatment outcome"
    assert row["Area"] == "Gastroenterology"
    assert row["Modality"] == "Bulk RNAseq"
    assert row["Predicts"] == (
        "Identify patients who achieve endoscopic remission after adalimumab treatment."
    )


def test_tasks_table_is_ordered_by_task_id(registry):
    assert next(iter(tasks_table(list(registry.values()))["Task"])) == (
        "IBD adalimumab remission"
    )


def test_tasks_table_never_leaks_an_accession(registry):
    rendered = tasks_table(list(registry.values())).to_string()
    assert not re.search(r"GSE\d+|E-MTAB-\d+|benchmark_", rendered)


def test_source_repositories_name_the_archive_not_the_study(registry):
    repositories = source_repositories(list(registry.values()))
    assert repositories == ["EMBL-EBI ArrayExpress", "NCBI GEO"]
    assert not re.search(r"GSE\d+|E-MTAB-\d+", " ".join(repositories))


def test_the_unvalidated_licence_field_never_reaches_a_public_page(registry):
    """One hand-written line per dataset must not read as a checked rights position."""
    assert "license" in next(iter(registry.values()))
    assert not any(
        "Terms of Use" in r for r in source_repositories(list(registry.values()))
    )


def test_empty_registry_yields_an_empty_table_not_a_crash():
    assert list(tasks_table([]).columns) == TASK_COLUMNS


def test_one_name_holds_results_for_two_modalities_at_once(results):
    """Their tasks are disjoint, so neither submission displaces the other."""
    registry = {tid: dict(task) for tid, task in REGISTRY.items()}
    registry["t900"] = {
        **registry["t901"],
        "task_id": "t900",
        "dataset_id": "d917",
        "cohort_id": "cohort-e",
        "modality": "Single Cell RNA",
        "title": "COVID severity",
    }
    boards = build_boards(registry)
    bulk = next(b for b in boards if b.modality == "bulk RNA")
    single = next(b for b in boards if b.modality == "Single Cell RNA")
    df = results(
        *_full("m", "2026-01-01"),
        ("m", "t900", 0.42, "2026-02-01"),
    )

    assert list(ranked_table(df, registry, bulk)["Model"]) == ["m"]
    single_board = ranked_table(df, registry, single)
    assert list(single_board["Model"]) == ["m"]
    assert single_board.loc[0, "Treatment outcome (AUROC)"] == pytest.approx(0.42)


def test_coverage_accumulates_across_separate_submissions(registry, named, results):
    rows = _full("m", "2026-01-01")
    first = results(*rows[:3]).assign(submission_id="first")
    second = results(*rows[3:]).assign(submission_id="second")
    second["submitted_at"] = "2026-02-01"

    board = ranked_table(pd.concat([first, second]), registry, named("bulk RNA"))

    assert list(board["Model"]) == ["m"]


def test_correcting_one_task_leaves_the_other_slots_standing(registry, named, results):
    df = results(
        *_full("m", "2026-01-01"),
        ("m", "t901", 0.11, "2026-02-01"),
    )

    board = ranked_table(df, registry, named("bulk RNA"))

    assert board.loc[0, "Treatment outcome (AUROC)"] == pytest.approx(0.11)
    assert board.loc[0, "Endotype (AUROC)"] == pytest.approx(ALL_SCORES[4])
    assert len(latest_only(df)) == len(REGISTRY)


def test_building_coverage_task_by_task_is_not_counted_as_resubmitting(results):
    """Twenty tasks over twenty submissions is coverage, not re-rolling: max is 1."""
    df = results(
        *[("m", tid, 0.5, f"2026-01-{i + 1:02d}") for i, tid in enumerate(REGISTRY)]
    )

    assert submission_counts(df) == {("m", False): 1}


def test_resubmitting_one_task_counts_that_slot_not_the_whole_board(results):
    df = results(
        *_full("m", "2026-01-01"),
        ("m", "t901", 0.2, "2026-02-01"),
        ("m", "t901", 0.3, "2026-03-01"),
    )

    assert submission_counts(df) == {("m", False): 3}


def test_the_submissions_count_column_reports_an_integer(registry, named, results):
    df = results(*_full("m", "2026-01-01"), ("m", "t901", 0.2, "2026-02-01"))

    board = ranked_table(df, registry, named("bulk RNA"))

    assert board.loc[0, "Submissions count"] == 2
    assert pd.api.types.is_integer_dtype(board["Submissions count"])


def test_a_baseline_shows_no_submission_count(registry, named, flagged_results):
    """We republish baselines on a schedule; the number would measure that, not tuning."""
    df = flagged_results(
        *_flagged("HVG-1200-genes", "2026-01-01", True),
        *_flagged("HVG-1200-genes", "2026-02-01", True),
    )

    board = ranked_table(df, registry, named("bulk RNA"))

    assert pd.isna(board.loc[0, "Submissions count"])


def test_an_hvg_entry_shows_no_submission_count(registry, named, results):
    """Each ``+ HVG`` re-sync appends rows; the number would measure our releases."""
    df = results(*_full("m + HVG", "2026-01-01"), *_full("m + HVG", "2026-02-01"))

    board = ranked_table(df, registry, named("bulk RNA"))

    assert pd.isna(board.loc[0, "Submissions count"])


def test_credits_come_from_the_most_recent_of_the_accumulated_submissions(
    registry, named, results
):
    """A later submission can add a paper link without resending every task."""
    rows = _full("m", "2026-01-01")
    first = results(*rows[:3]).assign(
        submission_id="first", institution="Old Lab", paper_link=""
    )
    second = results(*rows[3:]).assign(
        submission_id="second",
        institution="New Lab",
        paper_link="https://example.org/p",
    )
    second["submitted_at"] = "2026-02-01"

    board = ranked_table(pd.concat([first, second]), registry, named("bulk RNA"))

    assert board.loc[0, "Institution"] == "New Lab"
    assert board.loc[0, "Model"].paper_link == "https://example.org/p"


def test_correcting_one_task_with_blank_credits_keeps_the_earlier_ones(
    registry, named, results
):
    """A one-task fix resubmits with the credit fields blank; it must not wipe them."""
    full = results(*_full("m", "2026-01-01")).assign(
        institution="Lab A",
        paper_link="https://example.org/p",
        is_author_submission=True,
    )
    fix = results(("m", "t901", 0.11, "2026-02-01")).assign(
        institution="", paper_link="", is_author_submission=False
    )

    board = ranked_table(pd.concat([full, fix]), registry, named("bulk RNA"))

    assert board.loc[0, "Institution"] == "Lab A"
    assert board.loc[0, "Model"].paper_link == "https://example.org/p"
    assert board.loc[0, "Model"].is_author_submission is True


def test_an_entry_without_repeats_does_not_shrink_the_board_to_one_split(
    registry, named, flagged_results
):
    """A rival stored without repeats must not cost you your other splits.

    ``challenger`` is behind on split 0 and ahead on the two others. Keying
    battles by ``task:repeat`` and intersecting collapsed the board onto split 0.
    """
    board = named("bulk RNA")
    df = flagged_results(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_flagged("challenger", "2026-01-01", False, scores=(0.5667,) * 5),
    ).assign(
        repeat_scores=lambda frame: frame["model_name"].map(
            {"challenger": "[0.4, 0.6, 0.7]"}
        )
    )

    assert _elo(df, registry, board)["challenger"] > 1000


def test_an_entry_without_repeats_plays_its_mean_on_every_split():
    """The shorter record plays its mean; the opponent keeps all of its splits."""
    left, right = _paired_splits((0.4, 0.6, 0.7), (0.5,))

    assert left == (0.4, 0.6, 0.7)
    assert right == (0.5, 0.5, 0.5)


def test_splits_of_one_task_share_one_vote(registry, named, flagged_results):
    """``even`` wins six splits of t901 and loses one split of four other tasks.

    Weighted per split that is 6 wins against 4; weighted per task, 1 against 4.
    """
    board = named("bulk RNA")
    df = flagged_results(
        *_anchor("2026-01-01", scores=(0.5,) * 5),
        *_flagged("even", "2026-01-01", False, scores=(0.9, 0.1, 0.1, 0.1, 0.1)),
    ).assign(
        repeat_scores=lambda frame: [
            "[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]"
            if row.model_name == "even" and row.task_id == "t901"
            else ""
            for row in frame.itertuples()
        ]
    )

    assert _elo(df, registry, board)["even"] < 1000


def test_task_tables_flag_only_the_public_tasks(registry, named, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    for table in (
        per_task_table(df, registry, named("bulk RNA")),
        tasks_table(list(registry.values())),
    ):
        public = {str(cell) for cell in table["Task"] if cell.is_public}
        assert public == {"Psoriasis PASI"}
