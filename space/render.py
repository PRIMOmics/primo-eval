"""The PRIMO pages, rendered as ``pm-*`` markup for ``gr.HTML`` blocks.

Gradio has no card or rich-table component, so every page here is a string of
HTML styled entirely by ``primo.css`` -- the rail, the board grid, the
leaderboard tables and the tasks table. The functions are pure: they take the
data the app already fetched (``Board`` objects, the results ``DataFrame``, the
registry) and return a string, so they unit-test without Gradio or a network.

Two rules hold everywhere:

* Every string that comes from the registry or from a submission goes through
  ``html.escape`` -- board names, blurbs, disease names and, above all, the
  user-chosen model names that become table cells and column headers.
* New submissions can carry a public institution, author-submission check and paper
  link. Private legacy ``hf_username`` values stay hidden. Public strings and
  links are validated and escaped before rendering.

Navigation is reload-based: the rail is a column of ``<a href="?board=slug">``
and ``<a href="?tab=name">`` links, which costs a page load per click but buys
shareable per-board URLs and needs no JavaScript, matching how the board cards
have always worked. ``&view=private`` switches to the private view, which leaves
the public tasks out; every link carries it, so the view survives navigation.
"""

import math
import numbers
from html import escape
from urllib.parse import urlparse

import pandas as pd
from boards import (
    AREA_GROUP,
    CATEGORY_GROUP,
    GROUP_NOTE,
    MODALITY_GROUP,
    PUBLIC,
    Board,
    OpenBoard,
    board_label,
    in_group,
    modality_label,
    open_in_group,
    without_public,
)
from evaluator import _norm_id
from leaderboard import (
    ModelCell,
    TaskCell,
    per_task_table,
    ranked_table,
    tasks_table,
    top_models,
)
from results import without_hvg

SECTIONS = (MODALITY_GROUP, AREA_GROUP, CATEGORY_GROUP)
MEDALS = ("🥇", "🥈", "🥉")
N_TOP_MODELS = len(MEDALS)

VIEW_PARAM = "view"
PRIVATE_VIEW = "private"
VIEWS = (("All tasks", False), ("Private tasks", True))
ALL_PUBLIC = "Every task of this board is public."

BRAND = (
    '<a class="pm-brand" target="_self" href="{href}">'
    '<img src="/gradio_api/file=assets/primo-mark.webp" alt="">'
    "<b>PRIMO</b><span>benchmark</span></a>"
)

FOOT_LINKS = (
    ("tasks", "Tasks"),
    ("submit", "Submit a model"),
    ("contribute", "Contribute"),
    ("method", "Method"),
)

NAME_COLUMNS = frozenset({"Model", "Best"})
STRONG_COLUMNS = frozenset({"Task"})
METRIC_COLUMNS = frozenset({"Metric"})
RIGHT_COLUMNS = frozenset({"Rank", "Patients"})
METRIC_GUIDES = {
    "AUROC": "Range: 0 to 1 · Random predictor: 0.5",
    "Pearson": "Range: -1 to 1 · Random predictor: 0 (expected)",
    "Residual Spearman score": "Range: -1 to 1 · Random predictor: 0 (expected)",
    "Centered Spearman score": "Range: -1 to 1 · Random predictor: 0 (expected)",
}
PUBLIC_CHIP = (
    ' <span class="pm-public-chip" title="Labels downloadable from the PRIMO '
    'dataset" aria-label="Public labels">Public</span>'
)


def evaluation_confirmation() -> str:
    """Render the confirmation shown before a model evaluation starts."""
    return (
        '<dialog id="pm-evaluation-confirmation" class="pm-dialog" '
        'aria-labelledby="pm-evaluation-confirmation-title">'
        '<form method="dialog" class="pm-dialog-surface">'
        '<p class="pm-over">Confirm submission</p>'
        '<h2 id="pm-evaluation-confirmation-title">'
        "Evaluate and submit this model?</h2>"
        "<p>Evaluation can take several minutes. If it succeeds, this submission "
        "will update the public leaderboards.</p>"
        '<div class="pm-dialog-actions">'
        '<button class="pm-btn pm-btn--outline" value="cancel">Cancel</button>'
        '<button class="pm-btn pm-btn--primary" value="confirm">'
        "Evaluate model</button>"
        "</div></form></dialog>"
    )


def evaluation_status() -> str:
    """Render the progress message shown while an evaluation runs."""
    return (
        '<div class="pm-evaluation-status" role="status" aria-live="polite">'
        '<svg class="pm-spinner" aria-hidden="true" viewBox="0 0 24 24">'
        '<circle cx="12" cy="12" r="9"></circle>'
        '<path d="M12 3a9 9 0 0 1 9 9"></path></svg>'
        "<span>Your embeddings are being evaluated... This can take a few "
        "minutes.</span></div>"
    )


CHEVRON = (
    '<svg class="pm-chevron" viewBox="0 0 16 16" width="16" height="16" '
    'aria-hidden="true"><path d="M6 4l4 4-4 4" fill="none" stroke="#fff" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


# --------------------------------------------------------------------- view
def _href(query: str, private: bool) -> str:
    """A link to ``?query`` that keeps the private view on."""
    view = f"&{VIEW_PARAM}={PRIVATE_VIEW}" if private else ""
    return escape(f"?{query}{view}", quote=True)


def _in_view(board: Board, by_id: dict[str, dict], private: bool) -> Board:
    """The board as the current view shows it."""
    return without_public(board, by_id) if private else board


def _view_switch(query: str, private: bool) -> str:
    """The All tasks / Private tasks toggle of the page at ``?query``."""
    links = []
    for label, view in VIEWS:
        active = ' class="pm-active" aria-current="page"' if view == private else ""
        links.append(
            f'<a{active} target="_self" href="{_href(query, view)}">{label}</a>'
        )
    return (
        '<nav class="pm-toggle" aria-label="Tasks included" title="Private tasks '
        f'leave out the tasks whose labels are public">{"".join(links)}</nav>'
    )


# --------------------------------------------------------------------- rail
def rail_html(
    boards: list[Board],
    by_id: dict[str, dict],
    active_slug: str | None,
    active_tab: str | None,
    private: bool,
) -> str:
    """The left navigation: brand, a collapsible "Boards" menu, then the foot links.

    The board groups live inside a ``<details>`` toggle so the four foot links
    (Tasks, Submit, Contribute, Method) stay visible without scrolling. The toggle
    is pure HTML/CSS, keeping navigation reload-based and JavaScript-free.

    ``active_slug`` highlights the board a visitor is on; ``active_tab`` highlights
    a foot link, and its absence means we are on a board or the overview, so the
    "Boards" menu is the active one. Open boards link to Contribute, mirroring the
    overview cards. Counts follow the view.
    """
    on_boards = active_tab is None
    summary_cls = "pm-rail-summary pm-active" if on_boards else "pm-rail-summary"
    parts = [
        '<div class="pm-rail-inner">',
        BRAND.format(href=_href("tab=boards", private)),
        '<details class="pm-rail-section" open>',
        f'<summary class="{summary_cls}"><span>Boards</span>{CHEVRON}</summary>',
        '<div class="pm-rail-sections">',
    ]
    for group in SECTIONS:
        cards, opens = in_group(boards, group), open_in_group(boards, group)
        if not cards and not opens:
            continue
        parts.append(
            f'<div class="pm-rail-group"><p class="pm-rail-label">{escape(group)}</p>'
        )
        for board in cards:
            cls = "pm-link pm-active" if board.slug == active_slug else "pm-link"
            count = _in_view(board, by_id, private).n_tasks
            parts.append(
                f'<a class="{cls}" target="_self" '
                f'href="{_href(f"board={board.slug}", private)}">'
                f'<span class="pm-label">{escape(board_label(board))}</span>'
                f'<span class="pm-count">{count}</span></a>'
            )
        for board in opens:
            parts.append(
                '<a class="pm-link pm-link--open" target="_self" '
                f'href="{_href("tab=contribute", private)}">'
                f'<span class="pm-label">{escape(board.name)}</span>'
                '<span class="pm-badge pm-badge--quiet">open</span></a>'
            )
        parts.append("</div>")
    parts.append("</div></details>")
    parts.append('<div class="pm-rail-foot">')
    for tab, label in FOOT_LINKS:
        cls = "pm-foot-link pm-active" if tab == active_tab else "pm-foot-link"
        parts.append(
            f'<a class="{cls}" target="_self" href="{_href(f"tab={tab}", private)}">'
            f"{escape(label)}</a>"
        )
    parts.append("</div></div>")
    return "".join(parts)


# ------------------------------------------------------------------ boards
def _leading(top) -> str:
    """The card's mini-ranking: a medal, a name and an Elo, one line each.

    An empty board asks for the first submission instead of naming a leader.

    Every listed model gets a medal, so ``MEDALS`` is what bounds the podium and
    ``N_TOP_MODELS`` is derived from it -- the two cannot drift into a rank with
    no medal to print.

    A baseline is listed like any other leader. The card marks nothing; the board
    page is where a rating is read against the ``(baseline)`` it anchors on.
    """
    parts = ['<p class="pm-over" style="margin-top:14px">Leading (Elo)</p>']
    if not top:
        parts.append(
            '<p class="pm-leader pm-leader--2">No ranked model yet. Be the first.</p>'
        )
        return "".join(parts)
    for medal, row in zip(MEDALS, top, strict=False):
        score = "n/a" if row.elo is None else str(row.elo)
        parts.append(
            f'<p class="pm-leader"><span class="pm-medal">{medal}</span>'
            f"{escape(row.name)}"
            f' <span class="pm-num--dim">{score}</span></p>'
        )
    return "".join(parts)


def _live_card(
    board: Board, df: pd.DataFrame, by_id: dict[str, dict], private: bool
) -> str:
    """A board's card: its numbers and leaders in the current view."""
    shown = _in_view(board, by_id, private)
    href = _href(f"board={board.slug}", private)
    if not shown.task_ids:
        return (
            f'<a class="pm-card pm-card--open" target="_self" href="{href}">'
            f"<h3>{escape(board_label(board))}</h3>"
            f'<p class="pm-asks">{ALL_PUBLIC}</p></a>'
        )
    top = top_models(df, by_id, shown, N_TOP_MODELS)
    return (
        f'<a class="pm-card pm-card--live" target="_self" href="{href}">'
        f"<h3>{escape(board_label(board))}</h3>"
        f'<p class="pm-meta">{shown.n_tasks} tasks · {shown.n_cohorts} cohorts<br>'
        f"{shown.n_patients:,} patients · {shown.n_diseases} diseases</p>"
        f"{_leading(top)}</a>"
    )


def _open_card(board: OpenBoard, private: bool) -> str:
    return (
        '<a class="pm-card pm-card--open" target="_self" '
        f'href="{_href("tab=contribute", private)}">'
        f"<h3>{escape(board.name)}</h3>"
        '<p class="pm-meta"><span class="pm-badge pm-badge--quiet">OPEN · no cohort yet</span></p>'
        f'<p class="pm-asks">{escape(board.blurb)}</p>'
        '<p class="pm-cta">Propose a cohort →</p></a>'
    )


def render_boards(
    boards: list[Board], df: pd.DataFrame, by_id: dict[str, dict], private: bool
) -> str:
    """The Boards overview: every board as a card, grouped by facet.

    Live cards link to their board and teaser their leaders; open cards state a
    gap and link to Contribute. A section counts its open slices so the page
    reads as showing its own gaps, not the registry as the whole territory.

    The cards are the full board list in both views, so a board whose tasks are
    all public stays in place in the private view and says so.
    """
    if not boards:
        return (
            '<div class="pm-body"><p class="pm-caption">The task registry is '
            "unavailable right now. Please retry in a moment.</p></div>"
        )
    out = [
        '<div class="pm-head"><div><h1>Leaderboards</h1>',
        "<p>PRIMO evaluates zero-shot representations of omics samples through "
        "drug-development-related tasks. Benchmarks are organized by data "
        "modality, therapeutic area, or task category.</p></div>",
        _view_switch("tab=boards", private),
        '</div><div class="pm-body">',
    ]
    for group in SECTIONS:
        cards, opens = in_group(boards, group), open_in_group(boards, group)
        if not cards and not opens:
            continue
        note = escape(GROUP_NOTE.get(group, ""))
        counter = f" · +{len(opens)} open" if opens else ""
        out.append(
            '<div class="pm-group"><div class="pm-group-head">'
            f'<p class="pm-over">{escape(group)}</p>'
            f'<p class="pm-note">{note}{counter}</p></div><div class="pm-grid">'
        )
        out += [_live_card(b, df, by_id, private) for b in cards]
        out += [_open_card(b, private) for b in opens]
        out.append("</div></div>")
    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------- html tables
def _fmt_value(value, is_score: bool) -> str | None:
    """Display text for one cell; ``None`` marks a blank (``n/a``) cell."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            return None
        return f"{float(value):.3f}" if is_score else str(value)
    return str(value)


def _model_cell_html(cell: ModelCell) -> str:
    """Render one model label, allowing only escaped HTTP(S) paper links."""
    label = escape(str(cell))
    parsed = urlparse(cell.paper_link)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        url = escape(cell.paper_link, quote=True)
        label = (
            f'<a class="pm-model-link" href="{url}" target="_blank" '
            f'rel="noopener noreferrer">{label}</a>'
        )
    hvg_chip = (
        ' <span class="pm-hvg-chip" title="Concatenated with the HVG-1200 baseline '
        'features" aria-label="Concatenated with the HVG-1200 baseline features">'
        "+ HVG</span>"
        if cell.is_hvg
        else ""
    )
    author_chip = (
        ' <span class="pm-author-chip" title="Submitted by the model authors" '
        'aria-label="Submitted by the model authors">✓ Authors</span>'
        if cell.is_author_submission
        else ""
    )
    return label + hvg_chip + author_chip


def _score_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns to format and bold -- ``Rank`` is an index, not a score."""
    return [c for c in df.select_dtypes("number").columns if c not in RIGHT_COLUMNS]


def _score_guide(col: str, row: pd.Series) -> str | None:
    """Return metric context for a native score cell, when identifiable."""
    metric = next((name for name in METRIC_GUIDES if f"({name})" in col), None)
    if metric is None and col not in {"Elo", "Mean rank", "Mean score"}:
        metric = str(row.get("Metric", ""))
    return METRIC_GUIDES.get(metric) if metric is not None else None


def _bold_cells(df: pd.DataFrame, score_cols: list[str], axis: int) -> set:
    """Which ``(row, col)`` cells hold the best value.

    ``axis=0`` bolds the best model per column (the ranked table, read down);
    ``axis=1`` bolds the best model per row (the per-task table, read across).
    """
    bold = set()
    if not score_cols:
        return bold
    if axis == 0:
        for col in score_cols:
            best = (
                df[col].min(skipna=True)
                if col == "Mean rank"
                else df[col].max(skipna=True)
            )
            if pd.notna(best):
                for idx, value in df[col].items():
                    if pd.notna(value) and value == best:
                        bold.add((idx, col))
    else:
        for idx, row in df.iterrows():
            present = {c: row[c] for c in score_cols if pd.notna(row[c])}
            if present:
                best = max(present.values())
                bold.update((idx, c) for c, v in present.items() if v == best)
    return bold


def _cell_class(col: str, is_score: bool, blank: bool) -> str:
    if col in NAME_COLUMNS:
        return "pm-name"
    if col in STRONG_COLUMNS:
        return "pm-strong"
    if col in METRIC_COLUMNS:
        return "pm-metric"
    if is_score or col in RIGHT_COLUMNS:
        return "pm-num pm-num--dim" if blank else "pm-num"
    return ""


def _task_tooltip(task: dict) -> str:
    """Build the expanded clinical context shown for a task name."""
    description = str(task.get("description") or "").strip().rstrip(".")
    diseases = (
        ", ".join(str(d) for d in task.get("diseases") or []) or "the listed cohort"
    )
    patients = task.get("n_subjects") or task.get("n_samples")
    patient_text = (
        f"{patients:,} patients"
        if isinstance(patients, int)
        else "an unspecified number of patients"
    )
    n_samples = task.get("n_samples")
    if (
        isinstance(n_samples, int)
        and isinstance(patients, int)
        and n_samples != patients
    ):
        patient_text += f" ({n_samples:,} collection samples)"
    modality = str(task.get("modality") or "omics")
    tissue = str(task.get("tissue") or "unspecified tissue")
    target = str(task.get("target") or task.get("title") or "the task target")
    return (
        f"{description}. Patients: {patient_text} with {diseases}; "
        f"input data: {modality} profiles from {tissue.lower()}; "
        f"outcome: {target}."
    )


def _df_to_table(
    df: pd.DataFrame,
    bold_axis: int | None,
    empty: str,
    task_guides: dict[str, dict] | None = None,
) -> str:
    """Render a DataFrame as a ``pm-table``, escaping every header and cell."""
    if df.empty:
        return f'<p class="pm-caption">{escape(empty)}</p>'
    repeat_sd = df.attrs.get("repeat_standard_deviations", {})
    df = df.copy(deep=False)
    df.attrs = {}
    score_cols = _score_columns(df)
    bold = _bold_cells(df, score_cols, bold_axis) if bold_axis is not None else set()

    head = []
    for index, col in enumerate(df.columns):
        align = (
            ' style="text-align:right"'
            if col in score_cols or col in RIGHT_COLUMNS
            else ""
        )
        head.append(
            f'<th class="pm-sort" data-sort-index="{index}" tabindex="0" '
            f'role="button" aria-sort="none" title="Sort by {escape(str(col))}"'
            f"{align}>{escape(str(col))}</th>"
        )
    body = []
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            is_score = col in score_cols
            text = _fmt_value(row[col], is_score)
            cls = _cell_class(col, is_score, text is None)
            weight = ' style="font-weight:700"' if (idx, col) in bold else ""
            guide = (
                _score_guide(str(col), row) if is_score and text is not None else None
            )
            shown = (
                _model_cell_html(row[col])
                if isinstance(row[col], ModelCell)
                else "n/a"
                if text is None
                else escape(text)
            )
            standard_deviation = repeat_sd.get((idx, col))
            uncertainty = ""
            if standard_deviation and text is not None:
                deviation, repeats = standard_deviation
                sd_text = f"{deviation:.3f}"
                uncertainty = (
                    f" Standard deviation across {repeats} repeats: {sd_text}."
                )
                shown += f'<span class="pm-sd" aria-hidden="true"> ± {sd_text}</span>'
            if guide:
                escaped_guide = escape(guide, quote=True)
                score_label = escape(f"Score {text}.{uncertainty} {guide}", quote=True)
                shown = (
                    '<span class="pm-score-guide" tabindex="0" '
                    f'aria-label="{score_label}" '
                    f'data-tooltip="{escaped_guide}">{shown}</span>'
                )
            if task_guides and col == "Task" and text is not None:
                task = next(
                    (
                        candidate
                        for candidate in task_guides.values()
                        if candidate.get("title") == text
                    ),
                    None,
                )
                if task:
                    tooltip = _task_tooltip(task)
                    escaped_tooltip = escape(tooltip, quote=True)
                    task_label = escape(f"{text}. {tooltip}", quote=True)
                    shown = (
                        '<span class="pm-task-guide" tabindex="0" '
                        f'aria-label="{task_label}" data-tooltip="{escaped_tooltip}">{shown}</span>'
                    )
            if isinstance(row[col], TaskCell) and row[col].is_public:
                shown += PUBLIC_CHIP
            sort_value = (
                f' data-sort-value="{escape(text, quote=True)}"'
                if is_score and text is not None
                else ""
            )
            cells.append(f'<td class="{cls}"{sort_value}{weight}>{shown}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="pm-table-wrap"><table class="pm-table" style="table-layout:auto">'
        f"<thead><tr>{''.join(head)}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


# ------------------------------------------------------------------- board
def _board_meta(board: Board) -> str:
    return (
        f"{escape(board.blurb)} · {board.n_tasks} tasks · {board.n_cohorts} cohorts · "
        f"{board.n_patients:,} patients · {escape(modality_label(board.modality))}"
    )


HVG_TOGGLE = (
    '<input type="checkbox" id="pm-hvg-toggle" class="pm-hvg-toggle" checked>'
    '<label class="pm-hvg-switch" for="pm-hvg-toggle">Include models '
    '<span class="pm-hvg-chip">+ HVG</span></label>'
)


def _board_tables(
    board: Board, df: pd.DataFrame, by_id: dict[str, dict]
) -> tuple[str, str]:
    """The ranked and per-task tables of ``board`` over the rows of ``df``."""
    ranked = _df_to_table(
        ranked_table(df, by_id, board),
        bold_axis=0,
        empty="No model has covered every scored task of this board yet. Be the first to submit.",
    )
    per_task = _df_to_table(
        per_task_table(df, by_id, board),
        bold_axis=1,
        empty="No submission has scored on this board yet.",
        task_guides=by_id,
    )
    return ranked, per_task


def _hvg_variants(with_hvg: str, plain: str) -> str:
    """Both renderings of a table; ``HVG_TOGGLE`` shows one, via ``primo.css``."""
    return (
        f'<div class="pm-with-hvg">{with_hvg}</div>'
        f'<div class="pm-without-hvg">{plain}</div>'
    )


def _board_body(board: Board, df: pd.DataFrame, by_id: dict[str, dict]) -> str:
    """The HVG switch, then the ranked leaderboard and the per-task table.

    When ``+ HVG`` entries score on the board, both tables are rendered with and without them,
    so hiding them refits Elo and re-ranks rather than just dropping rows. A
    CSS-only checkbox, ticked by default, picks which rendering is shown. It must
    stay a sibling that precedes the groups, as ``primo.css`` selects with ``~``.
    """
    ranked, per_task = _board_tables(board, df, by_id)
    on_board = df[df["task_id"].map(_norm_id).isin(list(board.task_ids))]
    toggle = ""
    if len(without_hvg(on_board)) < len(on_board):
        plain_ranked, plain_per_task = _board_tables(board, without_hvg(df), by_id)
        ranked = _hvg_variants(ranked, plain_ranked)
        per_task = _hvg_variants(per_task, plain_per_task)
        toggle = HVG_TOGGLE
    return (
        f"{toggle}"
        '<div class="pm-group"><div class="pm-group-head">'
        '<p class="pm-over pm-over--marine">Ranked</p>'
        '<p class="pm-note">We evaluate zero-shot representations of models, so '
        "performance should not be considered as the best we can obtain. Only "
        "models that covered every task scored by the "
        "HVG-1200 log-normalized linear baseline "
        "(<code>HVG-1200-genes</code>) are ranked. "
        "<code>+ HVG</code> rows concatenate a model with the baseline's "
        "features.</p>"
        f"</div>{ranked}</div>"
        '<div class="pm-group"><div class="pm-group-head">'
        '<p class="pm-over pm-over--marine">Per task</p>'
        '<p class="pm-note">Every submission, partial ones included. Read across a row. '
        "Scores show mean ± SD when repeat-level results are available.</p>"
        f"</div>{per_task}</div>"
    )


def render_board(
    board: Board | None, df: pd.DataFrame, by_id: dict[str, dict], private: bool
) -> str:
    """One board: title strip and view switch, the ranked then per-task tables."""
    if board is None:
        return (
            '<div class="pm-head"><div><h1>No board available</h1>'
            "<p>The task registry could not be loaded. Please retry shortly.</p>"
            "</div></div>"
        )
    shown = _in_view(board, by_id, private)
    body = (
        _board_body(shown, df, by_id)
        if shown.task_ids
        else f'<p class="pm-caption">{ALL_PUBLIC} Switch to All tasks to see it.</p>'
    )
    return (
        '<div class="pm-head"><div><p class="pm-over">Board</p>'
        f"<h1>{escape(board_label(board))}</h1>"
        f"<p>{_board_meta(shown)}</p></div>"
        f"{_view_switch(f'board={board.slug}', private)}</div>"
        f'<div class="pm-body">{body}</div>'
    )


# ------------------------------------------------------------------- tasks
def render_tasks(by_id: dict[str, dict], private: bool) -> str:
    """The Tasks page: one scannable row per target. Provenance is never shown."""
    df = tasks_table(
        [task for task in by_id.values() if not (private and task.get(PUBLIC))]
    )
    table = _df_to_table(
        df,
        bold_axis=None,
        empty="The task registry is unavailable right now.",
        task_guides=by_id,
    )
    return (
        '<div class="pm-head"><div><h1>Tasks</h1>'
        f"<p>{len(df)} targets. One fixed task probe reads each "
        "one out of your embedding; the cohorts stay anonymous, the biology does "
        "not.</p></div>"
        f"{_view_switch('tab=tasks', private)}</div>"
        f'<div class="pm-body">{table}</div>'
    )
