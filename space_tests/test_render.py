"""Unit tests for the PRIMO page markup (no network, no Gradio).

These guard the guarantees a redesign must not lose: every view is reachable,
open slices route to Contribute, registry and submission text is escaped, the
baseline/first-submission states read correctly, the leaderboard bolds the best
value, provenance never leaks, and public submission credits are safely escaped.
"""

import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import render
from boards import ALL, build_boards, view_board, without_public
from leaderboard import ELO_ANCHOR_MODEL
from render import (
    SPACE_URL,
    cta_html,
    evaluation_confirmation,
    evaluation_status,
    hero_html,
    history_html,
    nav_html,
    render_tasks,
    render_view,
    version_notice,
)
from versions import CURRENT

from .conftest import CLOSED


def _view(registry, df, area=ALL, task=ALL, show_hvg=True, name="Bulk RNAseq"):
    """The bulk view of ``area`` and ``task``, as the leaderboard page shows it."""
    board = view_board(registry, "bulk-rna", area, task)
    return render_view(board, df, registry, name, show_hvg)


def _ranked(html: str) -> str:
    return html.split('class="pm-more"')[0]


def _per_task(html: str) -> str:
    return html.split('class="pm-more"')[1]


def _first_row(html: str) -> str:
    return re.search(r"<tbody><tr[^>]*>(.*?)</tr>", _ranked(html)).group(1)


# ------------------------------------------------------------------ the view
def test_the_view_card_names_the_view_and_counts_its_tasks(registry, results):
    name = "Bulk RNAseq · Gastroenterology · All tasks"
    html = _view(registry, results(), area="gastroenterology", name=name)

    assert name in html
    assert "<b>3</b> tasks · <b>2</b> cohorts · <b>210</b> patients" in html


def test_registry_text_is_escaped_in_the_view_name(registry, results):
    html = _view(registry, results(), name='<img src=x onerror="alert(1)">')

    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_a_view_without_tasks_says_so(registry, results):
    html = _view(registry, results(), area="dermatology", task="endotype")

    assert "No task in this view yet" in html
    assert "<b>0</b> tasks" in html


def test_an_unreachable_registry_is_not_an_empty_view(results):
    html = render_view(view_board({}, "bulk-rna", ALL, ALL), results(), {}, "", True)

    assert "The task registry could not be loaded" in html
    assert "No task in this view yet" not in html


def test_an_empty_view_invites_the_first_submission(registry, results):
    assert "be the first" in _view(registry, results()).lower()


def test_a_view_keeps_its_modality_apart(registry, results):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "Single Cell RNA",
        "title": "Single-cell RA endotype",
    }
    board = view_board(registry, "single-cell-rna", ALL, "endotype")
    df = results(("m", "t905", 0.5, "2026-01-01"), ("m", "t906", 0.6, "2026-01-01"))

    html = render_view(board, df, registry, "Single-cell RNAseq", True)

    assert "Single-cell RA endotype" in html
    assert "RA blood endotype" not in html
    assert ">Modality<" not in html


# --------------------------------------------------------------- the ranking
def test_ranked_table_bolds_the_best_and_captions_full_coverage(
    registry, flagged_results
):
    df = flagged_results(
        *[("eva", tid, 0.9, "2026-01-01", False) for tid in registry],
        *[("foo", tid, 0.5, "2026-01-01", False) for tid in registry],
    )
    html = _view(registry, df)

    assert "are ranked" in html
    assert "eva" in html and "foo" in html
    assert "pm-best" in _ranked(html)


def test_ranked_table_bolds_the_lowest_mean_rank():
    df = pd.DataFrame(
        {
            "Elo": [1200, 1000],
            "Mean rank": [2.0, 1.0],
            "Endotype (AUROC)": [0.9, 0.8],
        }
    )

    bold = render._bold_cells(df, list(df.columns), axis=0)

    assert (1, "Mean rank") in bold
    assert (0, "Mean rank") not in bold
    assert (0, "Elo") in bold
    assert (0, "Endotype (AUROC)") in bold


def test_ranked_table_does_not_bold_the_largest_submissions_count():
    df = pd.DataFrame(
        {
            "Elo": [1200, 1000],
            "Submissions count": pd.array([1, 7], dtype="Int64"),
        }
    )

    bold = render._bold_cells(df, render._score_columns(df), axis=0)

    assert (0, "Elo") in bold
    assert (1, "Submissions count") not in bold


def test_the_ranking_has_no_mean_score_and_one_line_per_model(registry, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    ranked = _ranked(_view(registry, df))
    css = Path(render.__file__).with_name("primo.css").read_text()

    assert "Mean score" not in ranked
    assert "Mean rank" in ranked
    assert "#pm-site .pm-ranked td { white-space: nowrap; }" in css


def test_the_top_three_are_framed_until_the_table_is_sorted(registry, results):
    scores = (("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6))
    df = results(
        *[
            (name, tid, score, "2026-01-01")
            for name, score in scores
            for tid in registry
        ]
    )
    css = Path(render.__file__).with_name("primo.css").read_text()

    rows = re.findall(r'<tr class="([^"]*)">', _ranked(_view(registry, df)))

    assert rows == ["pm-top pm-top-first", "pm-top", "pm-top pm-top-last", ""]
    assert '.pm-ranked[data-sorted="true"] tr.pm-top td' in css


def test_data_tables_expose_sortable_column_headers(registry, results):
    df = results(*[("eva", task_id, 0.8, "2026-01-01") for task_id in registry])
    html = _view(registry, df)

    assert re.search(r'<th class="pm-sort[^"]*" data-sort-index="0"', html)
    assert 'aria-sort="none"' in html
    assert '<td class="pm-rank" data-sort-value="1">1st</td>' in html


def test_ranked_view_shows_only_the_institution_credit(registry, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["institution"] = "Example Lab"

    html = _view(registry, df)

    assert "Institution" in html and "Example Lab" in html
    assert "Team" not in html
    assert "Author" not in html


def test_view_escapes_submission_credit(registry, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["institution"] = "<script>alert(1)</script>"

    html = _view(registry, df)

    assert "<script>" not in html and "&lt;script&gt;" in html


def test_author_submission_gets_a_check_chip_and_paper_link(registry, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["is_author_submission"] = True
    df["paper_link"] = "https://example.org/paper"

    html = _view(registry, df)

    assert 'href="https://example.org/paper"' in html
    assert ">eva</a>" in html
    assert 'class="pm-tag pm-tag--author"' in html
    assert "✓ Authors" in html


def test_unsafe_paper_link_is_not_rendered_as_a_link(registry, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["paper_link"] = 'javascript:alert("x")'

    html = _view(registry, df)

    assert "javascript:" not in html
    assert ">eva<" in html


def test_view_keeps_legacy_private_owners_hidden(registry, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["hf_username"] = "legacy-private-owner"

    assert "legacy-private-owner" not in _view(registry, df)


def test_view_escapes_a_hostile_model_name(registry, flagged_results):
    hostile = "<img src=x onerror=1>"
    df = flagged_results(
        *[(hostile, tid, 0.9, "2026-01-01", False) for tid in registry]
    )

    html = _view(registry, df)

    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_a_baseline_reads_the_same_in_both_tables(registry, flagged_results):
    df = flagged_results(
        *[("pca-50", tid, 0.8, "2026-01-01", True) for tid in registry],
        *[("eva", tid, 0.9, "2026-01-01", False) for tid in registry],
    )
    chip = render.BASELINE_CHIP

    html = _view(registry, df)

    assert f'<span class="pm-lrn">pca-50</span>{chip}' in _ranked(html)
    assert f">pca-50 {chip}</th>" in _per_task(html)
    assert "pca-50 (baseline)</th>" not in html


def test_a_baseline_that_leads_a_task_shows_its_chip_in_best(registry, flagged_results):
    df = flagged_results(
        *[("pca-50", tid, 0.9, "2026-01-01", True) for tid in registry],
        *[("eva", tid, 0.8, "2026-01-01", False) for tid in registry],
    )

    per_task = _per_task(_view(registry, df))

    assert f'<td class="pm-name">pca-50 {render.BASELINE_CHIP}</td>' in per_task
    assert "pca-50 (baseline)</td>" not in per_task


def test_the_submissions_header_says_how_the_count_works(registry, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])

    ranked = _ranked(_view(registry, df))

    assert "the most times one task of this model was submitted" in ranked
    assert 'title="Sort by Subm. Submissions:' in ranked


# --------------------------------------------------------- the toggle pills
def test_hvg_entry_shows_its_parent_name_with_an_hvg_chip(registry, flagged_results):
    df = flagged_results(
        *[("eva + HVG", tid, 0.9, "2026-01-01", False) for tid in registry]
    )
    df["is_author_submission"] = True

    ranked = _ranked(_view(registry, df, area="gastroenterology"))

    assert re.search(
        r'<span class="pm-lrn">eva</span><span class="pm-tag pm-tag--hvg"[^>]*>'
        r'\+ HVG</span><span class="pm-tag pm-tag--author"',
        ranked,
    )


def test_hiding_hvg_entries_reranks_the_view(registry, flagged_results):
    df = flagged_results(
        *[("eva", tid, 0.5, "2026-01-01", False) for tid in registry],
        *[("eva + HVG", tid, 0.9, "2026-01-01", False) for tid in registry],
        *[("foo", tid, 0.7, "2026-01-01", False) for tid in registry],
    )

    shown = _view(registry, df, area="gastroenterology")
    hidden = _view(registry, df, area="gastroenterology", show_hvg=False)

    assert "pm-tag--hvg" in _first_row(shown)
    assert "pm-tag--hvg" not in _ranked(hidden)
    assert 'data-sort-value="foo"' in _first_row(hidden)
    assert ">eva + HVG</th>" in _per_task(shown)
    assert "eva + HVG" not in _per_task(hidden)


def _eva_skips_the_public_task(flagged_results, registry):
    """The anchor covers every task; ``eva`` skips the public ``t904``."""
    return flagged_results(
        *[(ELO_ANCHOR_MODEL, tid, 0.5, "2026-01-01", True) for tid in registry],
        *[("eva", tid, 0.8, "2026-01-01", False) for tid in registry if tid != "t904"],
    )


def test_hiding_public_tasks_ranks_a_model_that_skipped_only_them(
    registry, flagged_results
):
    df = _eva_skips_the_public_task(flagged_results, registry)
    board = view_board(registry, "bulk-rna", ALL, ALL)

    every = render_view(board, df, registry, "Bulk RNAseq", True)
    private = render_view(without_public(board, registry), df, registry, "", True)

    assert 'data-sort-value="eva"' not in _ranked(every)
    assert 'data-sort-value="eva"' in _ranked(private)
    assert "Psoriasis PASI" in _per_task(every)
    assert "Psoriasis PASI" not in _per_task(private)


def test_the_view_counts_follow_the_public_tasks_pill(registry, results):
    board = view_board(registry, "bulk-rna", ALL, ALL)

    every = render_view(board, results(), registry, "", True)
    private = render_view(
        without_public(board, registry), results(), registry, "", True
    )

    assert "<b>5</b> tasks · <b>4</b> cohorts · <b>350</b> patients" in every
    assert "<b>4</b> tasks · <b>3</b> cohorts · <b>270</b> patients" in private


# --------------------------------------------------------- the per-task table
def test_per_task_names_expose_expanded_accessible_context(registry, results):
    df = results(*[("eva", task_id, 0.8, "2026-01-01") for task_id in registry])
    html = _view(registry, df)

    assert 'class="pm-task-guide"' in html
    assert "Patients: 90 patients with Crohn Disease, Ulcerative Colitis" in html
    assert "input data: bulk RNA profiles from digestive system" in html
    assert "outcome: endoscopic remission" in html
    assert 'tabindex="0"' in html


def test_per_task_scores_show_sd_only_when_repeats_are_available(registry, results):
    df = results(
        ("repeated", "t901", 0.7, "2026-01-01"),
        ("single", "t901", 0.6, "2026-01-01"),
        ("missing", "t901", 0.5, "2026-01-01"),
    ).assign(repeat_scores=["[0.6, 0.8]", "[0.6]", ""])

    html = _view(registry, df, area="gastroenterology")

    assert '<span class="pm-sd" aria-hidden="true"> ± 0.100</span>' in html
    css = Path(render.__file__).with_name("primo.css").read_text()
    assert ".pm-table .pm-sd { display: inline;" in css
    assert 'data-sort-value="0.700"' in html
    assert "Standard deviation across 2 repeats: 0.100" in html
    assert html.count('class="pm-sd"') == 1
    assert "Scores show mean ± SD when repeat-level results are available." in html


def test_task_tooltip_discloses_repeated_collection_samples(registry, results):
    registry["t901"]["n_subjects"] = 20
    registry["t901"]["n_samples"] = 126

    html = _view(registry, results(("eva", "t901", 0.8, "2026-01-01")))

    assert "20 patients (126 collection samples)" in html


def test_native_score_cells_explain_ranges_and_random_performance(registry, results):
    df = results(*[("eva", task_id, 0.8, "2026-01-01") for task_id in registry])
    html = _view(registry, df)

    assert 'data-tooltip="Range: 0 to 1 · Random predictor: 0.5"' in html
    assert 'data-tooltip="Range: -1 to 1 · Random predictor: 0 (expected)"' in html
    assert re.search(
        r'<td class="pm-num"[^>]*><span class="pm-score-guide" tabindex="0" '
        r'aria-label="Score 0\.800\. Range: 0 to 1 · Random predictor: 0\.5"',
        html,
    )
    assert re.search(
        r'<td class="pm-num"[^>]*><span class="pm-score-guide" tabindex="0" '
        r'aria-label="Score 0\.800\. Range: -1 to 1 · Random predictor: 0 '
        r'\(expected\)"',
        html,
    )

    centered = render._df_to_table(
        pd.DataFrame(
            [
                {"Metric": "Centered Spearman score", "model": 0.2},
                {"Metric": "Centered Spearman score", "model": None},
            ]
        ),
        bold_axis=1,
        empty="",
    )
    assert 'data-tooltip="Range: -1 to 1 · Random predictor: 0 (expected)"' in centered
    assert "Score None" not in centered
    assert '<td class="pm-num pm-num--dim">n/a</td>' in centered


def test_only_public_tasks_get_a_public_chip(registry, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    for html in (_view(registry, df), render_tasks(registry)):
        assert html.count('class="pm-public-chip"') == 1
        assert re.search(r'Psoriasis PASI(</span>)+ <span class="pm-public-chip"', html)


# ------------------------------------------------------------------- tasks page
def test_tasks_page_shows_biology_but_never_an_accession(registry):
    html = render_tasks(registry)
    assert "Crohn Disease" in html
    assert "Treatment outcome" in html
    assert not re.search(r"GSE\d+|E-MTAB-\d+", html)


def test_tasks_page_escapes_registry_text(registry):
    registry["t904"]["diseases"] = ['<img src=x onerror="alert(1)">']
    html = render_tasks(registry)
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_pages_carry_no_view_switch(registry):
    for html in (render_tasks(registry), nav_html("tasks")):
        assert "pm-private-toggle" not in html
        assert "view=" not in html
    assert "Psoriasis PASI" in render_tasks(registry)


# ------------------------------------------------------- closing panel, credit
def test_the_closing_panel_names_the_open_slices_and_links_contribute(boards):
    html = cta_html(boards)

    assert "PRIMO is open to any omics layer and any disease area." in html
    assert "Still missing, for example: " in html
    assert "Proteomics" in html and "Neuroscience" in html
    assert 'href="?tab=contribute"' in html


def test_an_open_slice_disappears_once_the_registry_covers_it(registry):
    covered = {
        **registry,
        "t906": {**registry["t905"], "task_id": "t906", "modality": "Single Cell RNA"},
    }

    assert "Single-cell RNAseq" in cta_html(build_boards(registry))
    assert "Single-cell RNAseq" not in cta_html(build_boards(covered))


def test_the_footer_credits_strable_and_ships_its_license():
    license_text = Path(render.__file__).with_name("LICENSE-STRABLE").read_text()

    assert "STRABLE" in render.CREDIT
    assert license_text.startswith("BSD 3-Clause License")
    assert "Copyright (c) 2026 The STRABLE developers." in license_text


# ---------------------------------------------------- leaderboard versions
def test_the_current_version_shows_no_closed_notice():
    assert version_notice(CURRENT) == ""
    assert "pm-notice" not in nav_html("leaderboard")


def test_a_closed_version_says_so_under_the_top_bar(monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])

    html = nav_html("leaderboard", CLOSED)

    assert html.index("</header>") < html.index('class="pm-notice"')
    assert f"from {CLOSED.opened} to {CURRENT.opened}" in html
    assert (
        f'href="?tab=leaderboard">Go to the current version, v{CURRENT.number}' in html
    )


def test_a_closed_version_keeps_every_top_bar_link_on_it_but_submit(monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])
    tag = f"&amp;version={CLOSED.number}"

    html = nav_html("tasks", CLOSED)

    for tab in ("leaderboard", "tasks", "method", "contribute"):
        assert f'href="?tab={tab}{tag}"' in html
    assert 'href="?tab=submit"' in html


def test_the_hero_names_the_version_on_screen(boards, results, monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])

    current = hero_html(boards, results())
    closed = hero_html(boards, results(), CLOSED)

    assert f'<span class="pm-ver">v{CURRENT.number}</span>' in current
    assert f'class="pm-ver pm-ver--closed">v{CLOSED.number} · closed' in closed
    assert 'href="#pm-history">Release notes</a>' in current


def test_the_hero_shows_how_a_model_is_scored(boards, results):
    html = hero_html(boards, results())

    assert 'class="pm-hero-art"' in html
    for step in ("Patient omics", "Your model", "Embedding", "Clinical tasks"):
        assert step in html
    assert "placeholder" not in html.lower()


def test_the_release_notes_are_on_the_board_not_on_github(results, monkeypatch):
    noted = replace(CLOSED, notes="Two tasks added.")
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, noted])

    html = history_html(results())

    assert CURRENT.notes in html
    assert "Two tasks added." in html
    assert f"{noted.opened} to {CURRENT.opened}" in html
    assert "github.com" not in html


def test_the_release_notes_link_every_other_version_to_its_board(results, monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])
    tag = f"&amp;version={CLOSED.number}"

    on_current = history_html(results())
    on_closed = history_html(results(), CLOSED)

    assert f'href="?tab=leaderboard{tag}">View this board' in on_current
    assert 'href="?tab=leaderboard">View this board' not in on_current
    assert 'href="?tab=leaderboard">View this board' in on_closed


def test_a_rank_citation_skips_the_date_of_a_board_without_results(results):
    html = history_html(results())

    assert f"PRIMO leaderboard v{CURRENT.number}, {SPACE_URL}" in html


def test_the_cite_window_names_the_version_on_screen(monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])

    html = nav_html("tasks", CLOSED)

    assert f"note         = {{Leaderboard version {CLOSED.number}}}" in html
    assert f"PRIMO leaderboard v{CLOSED.number}</code>" in html


def test_the_cite_button_is_in_the_top_bar_not_under_the_ranking(registry, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])

    nav = nav_html("tasks")

    assert 'id="pm-cite-btn"' in nav and 'id="pm-cite-dialog"' in nav
    assert "Evaluating a model on PRIMO" not in _view(registry, df)


def test_a_version_that_closed_the_day_it_opened_says_so(monkeypatch):
    same_day = replace(CLOSED, opened=CURRENT.opened)
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, same_day])

    assert f"from {CURRENT.opened} to {CURRENT.opened}" in version_notice(same_day)


def test_the_contribute_link_keeps_a_closed_version(boards, monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])

    html = cta_html(boards, CLOSED)

    assert f'href="?tab=contribute&amp;version={CLOSED.number}"' in html


# ------------------------------------------------------ evaluation confirmation
def test_evaluation_confirmation_requires_an_explicit_choice():
    html = evaluation_confirmation()

    assert '<dialog id="pm-evaluation-confirmation"' in html
    assert 'aria-labelledby="pm-evaluation-confirmation-title"' in html
    assert 'value="cancel"' in html
    assert 'value="confirm"' in html
    assert "Evaluate and submit this model?" in html
    assert "update the public leaderboards" in html


def test_both_windows_open_in_the_middle_of_the_screen():
    css = Path(render.__file__).with_name("primo.css").read_text()

    assert "#pm-site .pm-dialog { margin: auto !important; }" in css


def test_evaluation_status_announces_that_scoring_is_in_progress():
    html = evaluation_status()

    assert 'class="pm-evaluation-status"' in html
    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    assert 'class="pm-spinner" aria-hidden="true"' in html
    assert "<circle" in html
    assert "<path" in html
    assert "Your embeddings are being evaluated... This can take a few minutes." in html
