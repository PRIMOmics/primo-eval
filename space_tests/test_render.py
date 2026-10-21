"""Unit tests for the PRIMO page markup (no network, no Gradio).

These guard the guarantees a redesign must not lose: every board is reachable,
open slices route to Contribute, registry and submission text is escaped, the
baseline/first-submission states read correctly, the leaderboard bolds the best
value, provenance never leaks, and public submission credits are safely escaped.
"""

import re
from pathlib import Path

import pandas as pd
import render
from boards import AREA_GROUP, GROUP_NOTE, MODALITY_GROUP, build_boards, open_in_group
from leaderboard import ELO_ANCHOR_MODEL
from render import (
    RELEASES_URL,
    SECTIONS,
    evaluation_confirmation,
    evaluation_status,
    rail_html,
    render_board,
    render_boards,
    render_tasks,
)
from versions import CURRENT

from .conftest import CLOSED


def _slugs(html: str) -> list[str]:
    return re.findall(r'href="\?board=([^"]+)"', html)


def _boards_html(boards, registry, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    return render_boards(boards, df, registry)


# ---------------------------------------------------------------- boards page
def test_every_board_is_reachable_from_the_boards_page(boards, registry, results):
    html = _boards_html(boards, registry, results)
    assert set(_slugs(html)) == {b.slug for b in boards}


def test_each_facet_section_shows_with_its_note(boards, registry, results):
    html = _boards_html(boards, registry, results)
    assert "Per therapeutic areas" in html
    assert "Task Category" in html
    assert "Every task of one omics layer" in html
    assert "PRIMO evaluates zero-shot representations of omics samples" in html


def test_modality_labels_start_with_a_capital(boards, registry, results):
    html = _boards_html(boards, registry, results)
    assert "Bulk RNAseq" in html
    assert "Single-cell RNAseq" in html
    assert "Proteomics" in html
    assert "Spatial transcriptomics" in html
    assert "Leaderboards" in html


def test_area_and_category_cards_name_their_modality(registry, results):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "Single Cell RNA",
    }
    html = _boards_html(build_boards(registry), registry, results)

    assert "Rheumatology - bulk RNAseq" in html
    assert "Rheumatology - scRNAseq" in html
    assert "Endotype - bulk RNAseq" in html
    assert "Endotype - scRNAseq" in html


def test_open_cards_state_the_slices_nobody_covers(boards, registry, results):
    html = _boards_html(boards, registry, results)
    assert "Proteomics" in html
    assert "Neuroscience" in html
    assert "OPEN" in html


def test_both_open_axes_get_their_own_counter(boards, registry, results):
    """Modality declares three gaps and Therapeutic Areas two; each has a counter."""
    html = _boards_html(boards, registry, results)
    assert f"{GROUP_NOTE[MODALITY_GROUP]} · +3 open" in html
    assert f"{GROUP_NOTE[AREA_GROUP]} · +2 open" in html


def test_an_open_card_sends_the_visitor_to_contribute(boards, registry, results):
    html = _boards_html(boards, registry, results)
    assert 'href="?tab=contribute"' in html
    assert "Propose a cohort" in html


def test_an_open_card_disappears_once_the_registry_covers_it(registry, results):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "Single Cell RNA",
    }
    boards = build_boards(registry)
    html = _boards_html(boards, registry, results)
    assert 'href="?board=single-cell-rna"' in html
    assert f"{GROUP_NOTE[MODALITY_GROUP]} · +2 open" in html
    assert "Proteomics" in html


def test_live_cards_teaser_the_leaders(boards, registry, results):
    html = _boards_html(boards, registry, results)
    assert "Leading" in html
    assert "eva" in html


def test_a_live_card_drops_the_prose_an_open_card_keeps(boards, registry, results):
    """Live cards are title, counts and podium; only open cards still explain."""
    html = _boards_html(boards, registry, results)
    assert "Every PRIMO task" not in html
    assert "No cohort yet" in html


def test_the_podium_is_medalled(boards, registry, results):
    """Rank reads off the medal, so the card needs no numbering column."""
    df = pd.concat(
        [
            results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry]),
            results(*[("scgpt", tid, 0.6, "2026-01-01") for tid in registry]),
        ],
        ignore_index=True,
    )
    html = render_boards(boards, df, registry)
    assert "🥇" in html
    assert "🥈" in html


def test_a_board_nobody_covered_invites_the_first_submission(boards, registry, results):
    """A board no model covered in full asks for the first entry, not a leader."""
    html = render_boards(boards, results(("eva", "t901", 0.8, "2026-01-01")), registry)
    assert "be the first" in html.lower()


def test_a_board_held_only_by_baselines_lists_its_models(
    boards, registry, flagged_results
):
    """A board card lists its leading models without special baseline copy."""
    df = flagged_results(
        *[("pca-50", tid, 0.8, "2026-01-01", True) for tid in registry]
    )
    html = render_boards(boards, df, registry)
    assert "be the first" not in html.lower()
    assert "pca-50" in html


def test_a_baseline_is_teased_like_any_other_leader(boards, registry, flagged_results):
    """The card tags nothing: a baseline is a podium line, not a footnote."""
    df = flagged_results(
        *[("pca-50", tid, 0.8, "2026-01-01", True) for tid in registry],
        *[("eva", tid, 0.9, "2026-01-01", False) for tid in registry],
    )
    html = render_boards(boards, df, registry)
    assert "pca-50" in html and "eva" in html
    assert "pm-badge--marine" not in html


def test_cards_carry_their_counts(boards, registry, results):
    html = _boards_html(boards, registry, results)
    assert "patients" in html and "cohorts" in html and "diseases" in html


def test_registry_text_is_escaped_on_the_boards_page(registry, results):
    """The board name is the registry text a card still prints, so it is the probe."""
    registry["t904"]["therapeutic_area"] = '<img src=x onerror="alert(1)">'
    boards = build_boards(registry)
    html = _boards_html(boards, registry, results)
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


# ------------------------------------------------------------------- board page
def test_ranked_table_bolds_the_best_and_captions_full_coverage(
    boards, registry, named, flagged_results
):
    df = flagged_results(
        *[("eva", tid, 0.9, "2026-01-01", False) for tid in registry],
        *[("foo", tid, 0.5, "2026-01-01", False) for tid in registry],
    )
    html = render_board(named("bulk RNA"), df, registry)
    assert "are ranked" in html
    assert "eva" in html and "foo" in html
    assert "font-weight:700" in html  # the best value is bolded


def test_ranked_table_bolds_the_lowest_mean_rank():
    df = pd.DataFrame(
        {
            "Elo": [1200, 1000],
            "Mean rank": [2.0, 1.0],
            "Mean score": [0.9, 0.8],
        }
    )

    bold = render._bold_cells(df, list(df.columns), axis=0)

    assert (1, "Mean rank") in bold
    assert (0, "Mean rank") not in bold
    assert (0, "Elo") in bold
    assert (0, "Mean score") in bold


def test_data_tables_expose_sortable_column_headers(boards, registry, named, results):
    df = results(*[("eva", task_id, 0.8, "2026-01-01") for task_id in registry])
    html = render_board(named("bulk RNA"), df, registry)
    assert 'class="pm-sort" data-sort-index="0"' in html
    assert 'aria-sort="none"' in html


def test_per_task_names_expose_expanded_accessible_context(registry, named, results):
    df = results(*[("eva", task_id, 0.8, "2026-01-01") for task_id in registry])
    html = render_board(named("bulk RNA"), df, registry)

    assert 'class="pm-task-guide"' in html
    assert "Patients: 90 patients with Crohn Disease, Ulcerative Colitis" in html
    assert "input data: bulk RNA profiles from digestive system" in html
    assert "outcome: endoscopic remission" in html
    assert 'tabindex="0"' in html


def test_per_task_scores_show_sd_only_when_repeats_are_available(
    registry, named, results
):
    df = results(
        ("repeated", "t901", 0.7, "2026-01-01"),
        ("single", "t901", 0.6, "2026-01-01"),
        ("missing", "t901", 0.5, "2026-01-01"),
    ).assign(repeat_scores=["[0.6, 0.8]", "[0.6]", ""])

    html = render_board(named("Gastroenterology"), df, registry)

    assert '<span class="pm-sd" aria-hidden="true"> ± 0.100</span>' in html
    css = Path(render.__file__).with_name("primo.css").read_text()
    assert ".pm-table .pm-sd { display: block;" in css
    assert 'data-sort-value="0.700"' in html
    assert "Standard deviation across 2 repeats: 0.100" in html
    assert html.count('class="pm-sd"') == 1
    assert "Scores show mean ± SD when repeat-level results are available." in html


def test_category_board_markup_is_scoped_to_its_named_modality(registry, results):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "Single Cell RNA",
        "title": "Single-cell RA endotype",
    }
    board = next(
        board
        for board in build_boards(registry)
        if board.name == "Endotype" and board.modality == "Single Cell RNA"
    )
    html = render_board(
        board,
        results(
            ("m", "t905", 0.5, "2026-01-01"),
            ("m", "t906", 0.6, "2026-01-01"),
        ),
        registry,
    )

    assert ">Area<" in html
    assert ">Modality<" in html
    assert ">Single-cell RNAseq<" in html
    assert ">Bulk RNAseq<" not in html
    assert "Single-cell RA endotype" in html
    assert "RA blood endotype" not in html
    assert "<h1>Endotype - scRNAseq</h1>" in html


def test_task_tooltip_discloses_repeated_collection_samples(registry, results):
    registry["t901"]["n_subjects"] = 20
    registry["t901"]["n_samples"] = 126
    boards = build_boards(registry)
    board = next(board for board in boards if board.name == "Bulk RNAseq")
    html = render_board(board, results(("eva", "t901", 0.8, "2026-01-01")), registry)
    assert "20 patients (126 collection samples)" in html


def test_native_score_cells_explain_ranges_and_random_performance(
    registry, named, results
):
    df = results(*[("eva", task_id, 0.8, "2026-01-01") for task_id in registry])
    html = render_board(named("bulk RNA"), df, registry)

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


def test_ranked_board_shows_only_the_institution_credit(
    boards, registry, flagged_results
):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["institution"] = "Scienta"
    html = render_board(build_boards(registry)[0], df, registry)
    assert "Institution" in html and "Scienta" in html
    assert "Team" not in html
    assert "Author" not in html


def test_board_escapes_submission_credit(registry, named, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["institution"] = "<script>alert(1)</script>"
    html = render_board(named("bulk RNA"), df, registry)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_author_submission_gets_a_check_chip_and_paper_link(
    registry, named, flagged_results
):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["is_author_submission"] = True
    df["paper_link"] = "https://example.org/paper"
    html = render_board(named("bulk RNA"), df, registry)
    assert 'href="https://example.org/paper"' in html
    assert ">eva</a>" in html
    assert 'class="pm-author-chip"' in html
    assert "✓ Authors" in html


VARIANT = re.compile(
    r'<div class="(pm-(?:with-hvg|without-hvg|all-tasks|private-tasks)[a-z -]*)">'
)


def _variants(html: str) -> dict[str, str]:
    """A board group's tables, one per switch state, keyed by their classes."""
    parts = VARIANT.split(html)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _ranked(html: str) -> dict[str, str]:
    return _variants(html.split("Per task")[0])


def _per_task(html: str) -> dict[str, str]:
    return _variants(html.split("Per task")[1])


def test_hvg_entry_shows_its_parent_name_with_an_hvg_chip(
    registry, named, flagged_results
):
    df = flagged_results(
        *[("eva + HVG", tid, 0.9, "2026-01-01", False) for tid in registry]
    )
    df["is_author_submission"] = True
    html = render_board(named("Gastroenterology"), df, registry)
    ranked = _ranked(html)["pm-with-hvg"].split('class="pm-hvg-chip"')
    assert len(ranked) == 2
    assert ranked[0].endswith("eva <span ")
    assert '+ HVG</span> <span class="pm-author-chip"' in ranked[1]


def test_hvg_entries_are_shown_by_default_and_can_be_hidden(
    registry, named, flagged_results
):
    df = flagged_results(
        *[("eva", tid, 0.5, "2026-01-01", False) for tid in registry],
        *[("eva + HVG", tid, 0.9, "2026-01-01", False) for tid in registry],
        *[("foo", tid, 0.7, "2026-01-01", False) for tid in registry],
    )
    html = render_board(named("Gastroenterology"), df, registry)
    assert render.HVG_TOGGLE in html
    assert re.search(r'<input type="checkbox"[^>]*pm-hvg-toggle" checked>', html)
    ranked = _ranked(html)
    with_hvg, plain = ranked["pm-with-hvg"], ranked["pm-without-hvg"]
    assert "pm-hvg-chip" in with_hvg and "pm-hvg-chip" not in plain
    rank_one = '<td class="pm-num">1</td><td class="pm-name">'
    assert f"{rank_one}eva <span" in with_hvg
    assert f"{rank_one}foo</td>" in plain
    per_task = _per_task(html)
    assert ">eva + HVG</th>" in per_task["pm-with-hvg"]
    assert "eva + HVG" not in per_task["pm-without-hvg"]


def test_switch_css_hides_the_unselected_variants():
    css = Path(render.__file__).with_name("primo.css").read_text()
    assert "#pm-main .pm-hvg-toggle:checked ~ .pm-group .pm-without-hvg" in css
    assert "#pm-main .pm-hvg-toggle:not(:checked) ~ .pm-group .pm-with-hvg" in css
    assert "#pm-main .pm-private-toggle:checked ~ .pm-group .pm-all-tasks" in css
    assert (
        "#pm-main .pm-private-toggle:not(:checked) ~ .pm-group .pm-private-tasks" in css
    )


def test_a_board_without_hvg_entries_has_no_toggle(registry, named, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    html = render_board(named("bulk RNA"), df, registry)
    assert "pm-hvg-toggle" not in html
    assert "pm-without-hvg" not in html


def test_hvg_entries_on_other_boards_add_no_toggle(registry, named, results):
    df = results(
        *[("eva", tid, 0.8, "2026-01-01") for tid in registry],
        ("eva + HVG", "t904", 0.9, "2026-01-01"),
    )
    gastro = render_board(named("Gastroenterology"), df, registry)
    derma = render_board(named("Dermatology"), df, registry)
    assert "pm-hvg-toggle" not in gastro
    assert "pm-hvg-toggle" in derma


def _eva_skips_the_public_task(flagged_results, registry):
    """The anchor covers every task; ``eva`` skips the public ``t904``."""
    return flagged_results(
        *[(ELO_ANCHOR_MODEL, tid, 0.5, "2026-01-01", True) for tid in registry],
        *[("eva", tid, 0.8, "2026-01-01", False) for tid in registry if tid != "t904"],
    )


def test_private_toggle_comes_unticked_below_the_hvg_one(registry, named, results):
    df = results(
        *[("eva", tid, 0.8, "2026-01-01") for tid in registry],
        ("eva + HVG", "t901", 0.9, "2026-01-01"),
    )
    html = render_board(named("bulk RNA"), df, registry)
    assert f"{render.HVG_TOGGLE}<br>{render.PRIVATE_TOGGLE}" in html
    assert re.search(r'<input type="checkbox"[^>]*pm-private-toggle">', html)
    assert set(_ranked(html)) == {
        f"{hvg} {scope}"
        for hvg in ("pm-with-hvg", "pm-without-hvg")
        for scope in ("pm-all-tasks", "pm-private-tasks")
    }


def test_only_private_tasks_ranks_a_model_that_skipped_only_public_ones(
    registry, named, flagged_results
):
    df = _eva_skips_the_public_task(flagged_results, registry)
    html = render_board(named("bulk RNA"), df, registry)
    ranked, per_task = _ranked(html), _per_task(html)
    eva_ranked = '<td class="pm-name">eva</td>'
    assert eva_ranked not in ranked["pm-all-tasks"]
    assert eva_ranked in ranked["pm-private-tasks"]
    assert "Psoriasis PASI" in per_task["pm-all-tasks"]
    assert "Psoriasis PASI" not in per_task["pm-private-tasks"]


def test_private_toggle_only_on_boards_mixing_public_and_private_tasks(
    registry, named, results
):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    assert "pm-private-toggle" in render_board(named("bulk RNA"), df, registry)
    for name in ("Gastroenterology", "Dermatology"):
        html = render_board(named(name), df, registry)
        assert "pm-private-toggle" not in html
        assert "pm-all-tasks" not in html


def test_pages_carry_no_view_switch(boards, registry, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    for html in (
        render_boards(boards, df, registry),
        render_tasks(registry),
        rail_html(boards, None, None),
    ):
        assert "pm-private-toggle" not in html
        assert "view=" not in html
    assert "Psoriasis PASI" in render_tasks(registry)


def test_unsafe_paper_link_is_not_rendered_as_a_link(registry, named, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["paper_link"] = 'javascript:alert("x")'
    html = render_board(named("bulk RNA"), df, registry)
    assert "javascript:" not in html
    assert ">eva<" in html


def test_board_keeps_legacy_private_owners_hidden(registry, named, flagged_results):
    df = flagged_results(*[("eva", tid, 0.9, "2026-01-01", False) for tid in registry])
    df["hf_username"] = "legacy-private-owner"
    html = render_board(named("bulk RNA"), df, registry)
    assert "legacy-private-owner" not in html


def test_board_escapes_a_hostile_model_name(registry, named, flagged_results):
    hostile = "<img src=x onerror=1>"
    df = flagged_results(
        *[(hostile, tid, 0.9, "2026-01-01", False) for tid in registry]
    )
    html = render_board(named("bulk RNA"), df, registry)
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_an_empty_board_invites_the_first_submission(registry, results, named):
    html = render_board(named("bulk RNA"), results(), registry)
    assert "be the first" in html.lower()


def test_a_missing_board_renders_a_message_not_a_crash(registry, results):
    html = render_board(None, results(), registry)
    assert "No board available" in html


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


# ------------------------------------------------------------------------ rail
def test_rail_links_every_board_and_marks_the_active_one(boards):
    active = boards[0].slug
    html = rail_html(boards, active, None)
    assert set(_slugs(html)) == {b.slug for b in boards}
    assert f'class="pm-link pm-active" target="_self" href="?board={active}"' in html
    assert "Submit a model" in html


def test_rail_boards_control_has_a_visible_chevron(boards):
    html = rail_html(boards, None, None)

    assert '<details class="pm-rail-section" open>' in html
    assert '<summary class="pm-rail-summary pm-active">' in html
    assert '<svg class="pm-chevron" viewBox="0 0 16 16" width="16" height="16"' in html
    assert 'stroke="#fff"' in html
    assert 'stroke-width="2"' in html

    css = Path(render.__file__).with_name("primo.css").read_text()
    chevron_rule = re.search(r"\.pm-rail-summary \.pm-chevron\s*\{([^}]*)\}", css)
    assert chevron_rule is not None
    assert "color: #fff" in chevron_rule.group(1)


def test_rail_groups_contain_their_board_links(boards):
    html = rail_html(boards, None, None)
    groups = re.findall(r'<div class="pm-rail-group">(.*?)</div>', html)
    grouped = "".join(groups)
    n_open = sum(len(open_in_group(boards, group)) for group in SECTIONS)

    assert set(_slugs(grouped)) == {board.slug for board in boards}
    assert grouped.count('href="?tab=contribute"') == n_open


def test_rail_open_slices_route_to_contribute(boards):
    html = rail_html(boards, None, None)
    assert 'href="?tab=contribute"' in html
    assert "Proteomics" in html


def test_rail_category_labels_render_in_uppercase():
    css = Path(render.__file__).with_name("primo.css").read_text()
    label_rule = re.search(r"\.pm-rail-label\s*\{([^}]*)\}", css)

    assert label_rule is not None
    assert "text-transform: uppercase" in label_rule.group(1)


def test_rail_category_labels_use_consistent_authored_casing(boards):
    html = rail_html(boards, None, None)
    labels = re.findall(r'<p class="pm-rail-label">([^<]+)</p>', html)

    assert labels == ["Per modality", "Per therapeutic areas", "Task Category"]


def _version_menu(html: str) -> str:
    return re.search(r'<details class="pm-version">(.*?)</details>', html).group(1)


def test_the_version_menu_sits_under_the_logo(boards):
    html = rail_html(boards, None, None)

    brand, menu = html.index('class="pm-brand"'), html.index('class="pm-version"')
    assert brand < menu < html.index('class="pm-rail-section"')


def test_the_version_menu_names_the_current_version_and_its_release_notes(boards):
    menu = _version_menu(rail_html(boards, None, None))

    assert f'Version v{CURRENT.number}</span><span class="pm-count">current' in menu
    assert f'href="{RELEASES_URL}/{CURRENT.release}"' in menu
    assert "version=" not in menu


def test_the_rail_of_the_current_version_has_no_closed_notice(boards):
    assert "pm-rail-notice" not in rail_html(boards, None, None)


def test_the_version_menu_marks_the_version_on_screen(boards, monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])
    menu = _version_menu(rail_html(boards, None, None, CLOSED))
    tag = f"&amp;version={CLOSED.number}"

    assert f'Version v{CLOSED.number}</span><span class="pm-count">closed' in menu
    assert f'class="pm-link pm-active" target="_self" href="?tab=boards{tag}"' in menu
    assert 'class="pm-link" target="_self" href="?tab=boards"' in menu
    assert f"from {CLOSED.opened}" in menu


def test_a_version_switch_stays_on_the_page_on_screen(boards, monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])
    slug, tag = boards[0].slug, f"&amp;version={CLOSED.number}"

    on_board = _version_menu(rail_html(boards, slug, None))
    on_tasks = _version_menu(rail_html(boards, None, "tasks"))
    on_submit = _version_menu(rail_html(boards, None, "submit"))

    assert f'href="?board={slug}{tag}"' in on_board
    assert f'href="?tab=tasks{tag}"' in on_tasks
    assert f'href="?tab=boards{tag}"' in on_submit


def test_a_closed_version_keeps_every_rail_link_on_it_but_submit(boards, monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])
    html = rail_html(boards, None, "tasks", CLOSED)
    tag = f"&amp;version={CLOSED.number}"

    assert set(_slugs(html.replace(tag, ""))) == {b.slug for b in boards}
    for slug in _slugs(html.replace(tag, "")):
        assert f'href="?board={slug}{tag}"' in html
    assert f'href="?tab=tasks{tag}"' in html
    assert 'href="?tab=submit"' in html
    assert f'href="{RELEASES_URL}/{CLOSED.release}"' in html


def test_a_closed_version_says_when_it_closed_and_links_the_current_one(
    boards, monkeypatch
):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])

    notice = re.search(
        r'<div class="pm-rail-notice">(.*?)</div>',
        rail_html(boards, None, None, CLOSED),
    ).group(1)

    assert f"from {CLOSED.opened} to {CURRENT.opened}" in notice
    assert f'href="?tab=boards">Go to the current version, v{CURRENT.number}' in notice


def test_the_cards_of_a_closed_version_keep_it(boards, registry, results, monkeypatch):
    monkeypatch.setattr(render, "VERSIONS", [CURRENT, CLOSED])
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    html = render_boards(boards, df, registry, CLOSED)
    tag = f"&amp;version={CLOSED.number}"

    assert {slug.removesuffix(tag) for slug in _slugs(html)} == {b.slug for b in boards}
    assert all(slug.endswith(tag) for slug in _slugs(html))
    assert f'href="?tab=contribute{tag}"' in html


# ------------------------------------------------------ evaluation confirmation
def test_evaluation_confirmation_requires_an_explicit_choice():
    html = evaluation_confirmation()

    assert '<dialog id="pm-evaluation-confirmation"' in html
    assert 'aria-labelledby="pm-evaluation-confirmation-title"' in html
    assert 'value="cancel"' in html
    assert 'value="confirm"' in html
    assert "Evaluate and submit this model?" in html
    assert "update the public leaderboards" in html


def test_evaluation_status_announces_that_scoring_is_in_progress():
    html = evaluation_status()

    assert 'class="pm-evaluation-status"' in html
    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    assert 'class="pm-spinner" aria-hidden="true"' in html
    assert "<circle" in html
    assert "<path" in html
    assert "Your embeddings are being evaluated... This can take a few minutes." in html


def test_only_public_tasks_get_a_public_chip(registry, named, results):
    df = results(*[("eva", tid, 0.8, "2026-01-01") for tid in registry])
    for html in (
        render_board(named("bulk RNA"), df, registry),
        render_tasks(registry),
    ):
        assert html.count('class="pm-public-chip"') == 1
        assert re.search(r'Psoriasis PASI</span> <span class="pm-public-chip"', html)
