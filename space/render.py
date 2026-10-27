"""The PRIMO pages, rendered as ``pm-*`` markup for ``gr.HTML`` blocks.

Gradio has no card or rich-table component, so every page here is a string of
HTML styled entirely by ``primo.css`` -- the top bar, the hero, the leaderboard
tables, the release notes and the tasks table. The functions are pure: they take
the data the app already fetched (``Board`` objects, the results ``DataFrame``,
the registry) and return a string, so they unit-test without Gradio or a network.

Two rules hold everywhere:

* Every string that comes from the registry or from a submission goes through
  ``html.escape`` -- board names, blurbs, disease names and, above all, the
  user-chosen model names that become table cells and column headers.
* New submissions can carry a public institution, author-submission check and paper
  link. Private legacy ``hf_username`` values stay hidden. Public strings and
  links are validated and escaped before rendering.

Pages are reload-based ``<a href="?tab=name">`` links in the top bar; the
leaderboard view is picked with Gradio pills in ``app.py``. On a closed board
version, every link adds ``&version=<number>`` so the visitor stays on it, except
Submit: a submission always goes to the current version.
"""

import math
import numbers
import re
from html import escape
from itertools import pairwise
from urllib.parse import urlparse

import pandas as pd
from boards import (
    AREA_GROUP,
    CATEGORY_GROUP,
    MODALITY_GROUP,
    Board,
    in_group,
    open_in_group,
)
from leaderboard import (
    ELO_ANCHOR_MODEL,
    ModelCell,
    TaskCell,
    per_task_table,
    ranked_table,
    tasks_table,
)
from results import (
    BASELINE_TAG,
    IS_BASELINE,
    MODEL_NAME,
    split_hvg_name,
    with_baseline_flag,
    without_hvg,
)
from versions import CURRENT, VERSIONS, Version

SECTIONS = (MODALITY_GROUP, AREA_GROUP, CATEGORY_GROUP)
VERSION_PARAM = "version"
NAME_COLUMNS = frozenset({"Model", "Best"})
STRONG_COLUMNS = frozenset({"Task"})
METRIC_COLUMNS = frozenset({"Metric"})
RIGHT_COLUMNS = frozenset({"Rank", "Patients", "Submissions count"})
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
BASELINE_CHIP = (
    '<span class="pm-tag pm-tag--base" title="Our reference model">Baseline</span>'
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


def _href(query: str, version: Version = CURRENT) -> str:
    """A reload link that stays on ``version``; the current one needs no tag."""
    if version != CURRENT:
        query = f"{query}&{VERSION_PARAM}={version.number}"
    return escape(f"?{query}")


def _number(version: Version) -> tuple[int, ...]:
    """The version number as integers, so ``0.10`` comes after ``0.9``."""
    return tuple(int(part) for part in version.number.split("."))


def _closed_on(version: Version) -> str:
    """The day the version after ``version`` opened."""
    later = [listed for listed in VERSIONS if _number(listed) > _number(version)]
    return min(later, key=_number).opened


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
    """One model on one line: its name, its chips, then a dotted leader.

    Only escaped HTTP(S) paper links become links.
    """
    label = escape(str(cell).removesuffix(f" {BASELINE_TAG}"))
    parsed = urlparse(cell.paper_link)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        url = escape(cell.paper_link, quote=True)
        label = (
            f'<a class="pm-model-link" href="{url}" target="_blank" '
            f'rel="noopener noreferrer">{label}</a>'
        )
    chips = []
    if cell.is_hvg:
        chips.append(
            '<span class="pm-tag pm-tag--hvg" title="Concatenated with the HVG-1200 '
            'baseline features" aria-label="Concatenated with the HVG-1200 baseline '
            'features">+ HVG</span>'
        )
    if cell.is_baseline:
        chips.append(BASELINE_CHIP)
    if cell.is_author_submission:
        chips.append(
            '<span class="pm-tag pm-tag--author" title="Submitted by the model '
            'authors" aria-label="Submitted by the model authors">✓ Authors</span>'
        )
    return (
        f'<div class="pm-pipe"><span class="pm-lrn">{label}</span>'
        f'{"".join(chips)}<span class="pm-leader"></span></div>'
    )


def _with_baseline_chip(name: str) -> str:
    """The escaped name, its ``(baseline)`` tag shown as the Baseline chip."""
    if not name.endswith(f" {BASELINE_TAG}"):
        return escape(name)
    return f"{escape(name.removesuffix(f' {BASELINE_TAG}'))} {BASELINE_CHIP}"


def _score_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns to format and bold -- ``Rank`` is an index, not a score."""
    return [c for c in df.select_dtypes("number").columns if c not in RIGHT_COLUMNS]


def _score_guide(col: str, row: pd.Series) -> str | None:
    """Return metric context for a native score cell, when identifiable."""
    metric = next((name for name in METRIC_GUIDES if f"({name})" in col), None)
    if metric is None and col not in {"Elo", "Mean rank"}:
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
        name = _with_baseline_chip(str(col))
        align = (
            ' style="text-align:right"'
            if col in score_cols or col in RIGHT_COLUMNS
            else ""
        )
        head.append(
            f'<th class="pm-sort" data-sort-index="{index}" tabindex="0" '
            f'role="button" aria-sort="none" title="Sort by {escape(str(col))}"'
            f"{align}>{name}</th>"
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
                else _with_baseline_chip(text)
                if col in NAME_COLUMNS
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
                        f'aria-label="{task_label}" data-tooltip="{escaped_tooltip}">'
                        f'<span class="pm-task-text">{shown}</span></span>'
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


# ------------------------------------------------------------------- tasks
def render_tasks(by_id: dict[str, dict]) -> str:
    """The Tasks page: one scannable row per target. Provenance is never shown."""
    df = tasks_table(list(by_id.values()))
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
        "not.</p></div></div>"
        f'<div class="pm-body">{table}</div>'
    )


# ------------------------------------------------------------- leaderboard
SPACE_URL = "https://huggingface.co/spaces/PRIMOmics/primo-eval"
DATA_URL = "https://huggingface.co/datasets/PRIMOmics/primo"
PAPER_URL = "https://openreview.net/forum?id=v2SA8gHwqo"

NAV_TABS = (
    ("leaderboard", "Leaderboard"),
    ("tasks", "Tasks"),
    ("method", "Method"),
    ("contribute", "Contribute"),
)
LEDE = (
    "PRIMO ranks omics foundation models on real "
    '<a target="_self" href="{tasks}">clinical tasks</a>. Your model turns each '
    "patient's omics profile into an embedding. One fixed probe, the same for every "
    "model, then predicts hidden outcomes such as treatment response or prognosis, "
    "so the score measures the embedding."
)
RANKED_NOTE = (
    '<p class="pm-note">We score zero-shot representations, so a score is not the '
    "best a model can reach. Only models that covered every task scored by the "
    f"<code>{ELO_ANCHOR_MODEL}</code> baseline are ranked. <code>+ HVG</code> rows "
    "concatenate a model with the baseline's features.</p>"
)
N_TOP = 3
ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}
NOT_SCORES = frozenset({"Rank", "Submissions count"})
RANKED_LABELS = {"Submissions count": "Subm."}
RANKED_NOTES = {"Model": "paper linked when given"}
RANKED_HINTS = {
    "Elo": f"The {ELO_ANCHOR_MODEL} baseline sits at 1000.",
    "Mean rank": "Lower is better.",
    "Submissions count": (
        "Submissions: the most times one task of this model was submitted."
    ),
}
SHORT_METRICS = {
    "Centered Spearman score": "centered Spearman",
    "Residual Spearman score": "residual Spearman",
}
CATEGORY_HEADER = re.compile(r"^(?P<name>.+) \((?P<metric>[^()]+)\)$")
STRABLE_URL = "https://gblayer.github.io/strable-leaderboard/"
BOARD_SUB = (
    '<p class="pm-board-sub">Models are ranked by Elo over the tasks of the view, '
    f"anchored on the {ELO_ANCHOR_MODEL} baseline at 1000. Task columns show the "
    "mean score, each in its own metric.</p>"
)
ART_GENES = (0.55, 0.9, 0.4, 0.75, 0.35, 0.95, 0.5, 0.8, 0.45, 0.7, 0.3, 0.85, 0.6, 0.4)
ART_EMBEDDING = (0.7, 1.0, 0.6, 0.9, 0.75, 0.95)
ART_LAYERS = ((4, 132), (5, 150), (3, 168))
ART_TASKS = ("Treatment response", "Prognosis", "Disease severity", "Molecular subtype")
CREDIT = (
    '<p class="pm-credit">Design inspired by the '
    f'<a href="{STRABLE_URL}" target="_blank" rel="noopener noreferrer">STRABLE '
    "leaderboard</a>.</p>"
)


def _art_text(x: float, y: float, text: str, cls: str, anchor: str = "middle") -> str:
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{text}</text>'


def _art_arrow(x1: float, x2: float, y: float) -> str:
    return (
        f'<line class="pm-art-line" x1="{x1}" y1="{y}" x2="{x2 - 6}" y2="{y}"></line>'
        f'<path class="pm-art-head" d="M{x2 - 7} {y - 4} L{x2} {y} L{x2 - 7} {y + 4} Z">'
        "</path>"
    )


def _hero_art() -> str:
    """The hero figure: a patient's omics, a model, one embedding, many tasks."""
    parts = [
        _art_text(58, 22, "Patient omics", "pm-art-h"),
        _art_text(162, 22, "Your model", "pm-art-h"),
        _art_text(254, 22, "Embedding", "pm-art-h"),
        _art_text(351, 22, "Clinical tasks", "pm-art-h"),
        '<circle class="pm-art-patient" cx="58" cy="44" r="8"></circle>',
        '<path class="pm-art-patient" d="M45 66 Q58 50 71 66 Z"></path>',
    ]
    parts += [
        f'<rect class="pm-art-{"b" if index % 3 == 0 else "a"}" x="47" '
        f'y="{74 + index * 11}" width="22" height="9" rx="1.5" opacity="{level}"></rect>'
        for index, level in enumerate(ART_GENES)
    ]
    parts += [
        _art_text(58, 246, "gene expression", "pm-art-c"),
        _art_arrow(78, 112, 150),
        '<rect class="pm-art-box" x="114" y="112" width="96" height="76" rx="14"></rect>',
    ]
    nodes = [
        [(162 + (k - (count - 1) / 2) * 16, y) for k in range(count)]
        for count, y in ART_LAYERS
    ]
    parts += [
        f'<line class="pm-art-edge" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"></line>'
        for upper, lower in pairwise(nodes)
        for x1, y1 in upper
        for x2, y2 in lower
    ]
    parts += [
        f'<circle class="pm-art-node" cx="{x}" cy="{y}" r="3.2"></circle>'
        for layer in nodes
        for x, y in layer
    ]
    parts += [
        _art_text(162, 204, "omics foundation model", "pm-art-c"),
        _art_arrow(212, 244, 150),
    ]
    parts += [
        f'<rect class="pm-art-{"a" if index % 2 else "b"}" x="246" y="{100 + index * 17}" '
        f'width="16" height="14" rx="2" opacity="{level}"></rect>'
        for index, level in enumerate(ART_EMBEDDING)
    ]
    parts += [
        _art_text(254, 218, "sample", "pm-art-c"),
        _art_text(254, 230, "representation", "pm-art-c"),
    ]
    for index, task in enumerate(ART_TASKS):
        y = 74 + index * 44
        parts += [
            f'<path class="pm-art-line" d="M268 150 C 278 150, 278 {y}, 288 {y}"></path>',
            f'<rect class="pm-art-pill" x="290" y="{y - 14}" width="122" height="28" '
            'rx="14"></rect>',
            _art_text(302, y + 4, task, "pm-art-t", anchor="start"),
        ]
    return (
        '<div class="pm-art"><svg class="pm-hero-art" viewBox="0 0 420 268" '
        'role="img" aria-labelledby="pm-art-title"><title id="pm-art-title">A patient\'s '
        "omics profile goes through your model, which returns one embedding per "
        "sample. The same fixed probe then scores that embedding on clinical tasks."
        f"</title>{''.join(parts)}</svg></div>"
    )


def updated_on(df: pd.DataFrame) -> str:
    """The day of the latest stored result, or ``""`` when there is none."""
    if df.empty or "submitted_at" not in df:
        return ""
    return str(df["submitted_at"].astype(str).max())[:10]


def _model_count(df: pd.DataFrame) -> int:
    """Submitted models, a ``+ HVG`` entry counted with its parent, baselines out."""
    flagged = with_baseline_flag(df)
    if flagged.empty:
        return 0
    return len(
        {
            split_hvg_name(str(model))[0]
            for model, is_baseline in zip(flagged[MODEL_NAME], flagged[IS_BASELINE])
            if not is_baseline
        }
    )


def version_notice(version: Version) -> str:
    """On a closed version, a banner that says so and links to the current one."""
    if version == CURRENT:
        return ""
    return (
        '<div class="pm-notice" role="status">'
        f"<b>Version v{escape(version.number)} is closed.</b> "
        f"Its scores, from {escape(version.opened)} to "
        f"{escape(_closed_on(version))}, are kept read-only. "
        f'<a target="_self" href="{_href("tab=leaderboard")}">'
        f"Go to the current version, v{escape(CURRENT.number)}</a></div>"
    )


def _cite_dialog(version: Version) -> str:
    """The window of the Cite button: how to cite PRIMO and the version on screen."""
    bibtex = (
        "@misc{primo2026,\n"
        "  title        = {{PRIMO}: Patient Representations in Multi-Omics},\n"
        "  author       = {{PRIMO contributors}},\n"
        "  year         = {2026},\n"
        f"  note         = {{Leaderboard version {version.number}}},\n"
        f"  howpublished = {{\\url{{{SPACE_URL}}}}}\n"
        "}"
    )
    return (
        '<dialog id="pm-cite-dialog" class="pm-dialog" aria-labelledby="pm-cite-title">'
        '<form method="dialog" class="pm-dialog-surface">'
        '<h2 id="pm-cite-title">Cite PRIMO</h2>'
        "<p><b>Evaluating a model on PRIMO?</b> Please cite PRIMO and give the board "
        "version you used, for example "
        f"<code>PRIMO leaderboard v{escape(version.number)}</code>.</p>"
        f"<pre>{escape(bibtex)}</pre>"
        '<div class="pm-dialog-actions">'
        '<button class="pm-btn pm-btn--outline" value="close">Close</button>'
        "</div></form></dialog>"
    )


def nav_html(active_tab: str, version: Version = CURRENT) -> str:
    """The top bar: the PRIMO mark, the pages, the data, the paper, Submit and Cite.

    On a closed version, every page link keeps it, except Submit, Cite names it,
    and a banner under the bar says the version is closed.
    """
    links = [
        f'<a class="{"pm-active" if tab == active_tab else ""}" target="_self" '
        f'href="{_href(f"tab={tab}", version)}">{name}</a>'
        for tab, name in NAV_TABS
    ]
    links += [
        f'<a href="{url}" target="_blank" rel="noopener noreferrer">{name}</a>'
        for url, name in ((DATA_URL, "Data"), (PAPER_URL, "Paper"))
    ]
    return (
        '<header class="pm-nav"><a class="pm-logo" target="_self" '
        f'href="{_href("tab=leaderboard", version)}" aria-label="PRIMO leaderboard">'
        '<img src="/gradio_api/file=assets/primo-mark.webp" alt="">'
        "<b>PRIMO</b></a>"
        '<div class="pm-nav-right">'
        f'<nav class="pm-nav-links" aria-label="Main">{"".join(links)}</nav>'
        '<a class="pm-btn" target="_self" href="?tab=submit">Submit a model</a>'
        '<button type="button" class="pm-btn pm-btn--outline" id="pm-cite-btn" '
        'aria-haspopup="dialog" aria-controls="pm-cite-dialog">Cite</button>'
        f"</div></header>{_cite_dialog(version)}{version_notice(version)}"
    )


def hero_html(boards: list[Board], df: pd.DataFrame, version: Version = CURRENT) -> str:
    """The title block: version, PRIMO, what it is, its size, and the visual."""
    layers = in_group(boards, MODALITY_GROUP)
    stats = (
        (sum(b.n_tasks for b in layers), "tasks"),
        (sum(b.n_cohorts for b in layers), "cohorts"),
        (sum(b.n_patients for b in layers), "patients"),
        (len(layers), "modalities"),
        (_model_count(df), "models"),
    )
    chips = "".join(f"<li><b>{n:,}</b> {name}</li>" for n, name in stats)
    updated = updated_on(df)
    stamp = (
        f'<span class="pm-updated">Updated {escape(updated)}</span>' if updated else ""
    )
    badge = f'<span class="pm-ver">v{escape(version.number)}</span>'
    if version != CURRENT:
        badge = (
            f'<span class="pm-ver pm-ver--closed">v{escape(version.number)} '
            "· closed</span>"
        )
    return (
        '<section class="pm-hero" aria-labelledby="pm-hero-title"><div>'
        '<p class="pm-eyebrow"><span class="pm-label">Leaderboard</span>'
        f'{badge}<a class="pm-ver-link" href="#pm-history">Release notes</a>'
        f"{stamp}</p>"
        '<h1 id="pm-hero-title">PRIMO</h1>'
        '<p class="pm-expand"><span class="pm-xp"><b>P</b>atient <b>R</b>epresentations '
        "<b>i</b>n <b>M</b>ulti-<b>O</b>mics</span></p>"
        f'<p class="pm-lede">{LEDE.format(tasks=_href("tab=tasks", version))}</p>'
        f'<ul class="pm-stats">{chips}</ul></div>'
        f'<div class="pm-art-wrap">{_hero_art()}</div>'
        "</section>"
    )


def _ordinal(rank: int) -> str:
    """``1`` -> ``1st``, ``12`` -> ``12th``."""
    suffix = "th" if 10 <= rank % 100 <= 20 else ORDINAL_SUFFIXES.get(rank % 10, "th")
    return f"{rank}{suffix}"


def _ranked_header(index: int, col: str, numeric: bool) -> str:
    match = CATEGORY_HEADER.match(col)
    if match:
        name = match["name"]
        note = SHORT_METRICS.get(match["metric"], match["metric"])
    else:
        name, note = RANKED_LABELS.get(col, col), RANKED_NOTES.get(col, "")
    align = " pm-r" if numeric else ""
    small = f"<small>{escape(note)}</small>" if note else ""
    hint = escape(
        f"Sort by {name.rstrip('.')}. {RANKED_HINTS.get(col, '')}".strip(), quote=True
    )
    return (
        f'<th class="pm-sort{align}" data-sort-index="{index}" tabindex="0" '
        f'role="button" aria-sort="none" title="{hint}">{escape(name)}{small}</th>'
    )


def _ranked_cell(
    table: pd.DataFrame, idx, col: str, bold: set, scores: list, numeric: set
) -> str:
    """One cell of the ranked table, on one line."""
    value = table.at[idx, col]
    if col == "Rank":
        return (
            f'<td class="pm-rank" data-sort-value="{int(value)}">'
            f"{_ordinal(int(value))}</td>"
        )
    if isinstance(value, ModelCell):
        return (
            f'<td class="pm-name" data-sort-value="{escape(str(value), quote=True)}">'
            f"{_model_cell_html(value)}</td>"
        )
    is_score = col in scores
    text = _fmt_value(value, is_score)
    if col not in numeric:
        shown = "n/a" if text is None else escape(text)
        dim = " pm-dim" if text is None else ""
        return f'<td class="pm-inst{dim}">{shown}</td>'
    if text is None:
        return '<td class="pm-num pm-dim">n/a</td>'
    shown = escape(text)
    guide = _score_guide(col, table.loc[idx]) if is_score else None
    if guide:
        label = escape(f"Score {text}. {guide}", quote=True)
        shown = (
            f'<span class="pm-score-guide" tabindex="0" aria-label="{label}" '
            f'data-tooltip="{escape(guide, quote=True)}">{shown}</span>'
        )
    cls = "pm-num"
    if col == "Elo":
        cls += " pm-key"
    if (idx, col) in bold:
        cls += " pm-best"
    return (
        f'<td class="{cls}" data-sort-value="{escape(text, quote=True)}">{shown}</td>'
    )


def _count(n: int, noun: str) -> str:
    """``<b>1</b> task``, ``<b>6</b> tasks``."""
    return f"<b>{n:,}</b> {noun}{'' if n == 1 else 's'}"


def _view_head(board: Board, view_name: str) -> str:
    """The head of the ranking card: the view, then how many tasks it holds."""
    counts = " · ".join(
        (
            _count(board.n_tasks, "task"),
            _count(board.n_cohorts, "cohort"),
            _count(board.n_patients, "patient"),
        )
    )
    return (
        f'<div class="pm-card-head"><span class="pm-view">{escape(view_name)}</span>'
        f'<span class="pm-view-n">{counts}</span></div>'
    )


def _ranked_html(table: pd.DataFrame) -> str:
    """The ranked table: one line per model, the top three framed."""
    if table.empty:
        return (
            '<p class="pm-empty">No model has covered every scored task of this view '
            "yet. Be the first to submit.</p>"
        )
    numeric = set(table.select_dtypes("number").columns)
    scores = [c for c in table.columns if c in numeric and c not in NOT_SCORES]
    bold = _bold_cells(table, scores, axis=0)
    head = "".join(
        _ranked_header(index, str(col), col in numeric)
        for index, col in enumerate(table.columns)
    )
    n_top = min(N_TOP, len(table))
    body = []
    for position, idx in enumerate(table.index):
        classes = []
        if position < n_top:
            classes.append("pm-top")
        if position == 0:
            classes.append("pm-top-first")
        if position == n_top - 1:
            classes.append("pm-top-last")
        cells = "".join(
            _ranked_cell(table, idx, str(col), bold, scores, numeric)
            for col in table.columns
        )
        body.append(f'<tr class="{" ".join(classes)}">{cells}</tr>')
    return (
        '<div class="pm-tbl-wrap"><table class="pm-table pm-ranked">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _per_task_details(per_task: pd.DataFrame, by_id: dict, board: Board) -> str:
    """The per-task scores, folded under the ranking."""
    table = _df_to_table(
        per_task,
        bold_axis=1,
        empty="No submission has scored on this view yet.",
        task_guides=by_id,
    )
    return (
        '<details class="pm-more"><summary><span class="pm-title">Scores per task'
        f'</span><span class="pm-more-n">{board.n_tasks} tasks</span></summary>'
        '<p class="pm-note">Every submission, partial ones included. Read across a '
        "row. Scores show mean ± SD when repeat-level results are available.</p>"
        f"{table}</details>"
    )


def render_view(
    board: Board,
    df: pd.DataFrame,
    by_id: dict[str, dict],
    view_name: str,
    show_hvg: bool,
) -> str:
    """One view: its size, its ranking, then its per-task scores."""
    rows = df if show_hvg else without_hvg(df)
    head = _view_head(board, view_name)
    if not by_id:
        return (
            f'<div class="pm-card">{head}<p class="pm-empty">The task registry could '
            "not be loaded. Please retry in a moment.</p></div>"
        )
    if not board.task_ids:
        return (
            f'<div class="pm-card">{head}<p class="pm-empty">No task in this view '
            "yet. Pick another therapeutic area or task.</p></div>"
        )
    ranked = _ranked_html(ranked_table(rows, by_id, board))
    return (
        f'<div class="pm-card">{head}{ranked}</div>'
        + RANKED_NOTE
        + _per_task_details(per_task_table(rows, by_id, board), by_id, board)
    )


def history_html(df: pd.DataFrame, version: Version = CURRENT) -> str:
    """The release notes: every board version, what changed, and its board.

    A version other than the one on screen links to its board, so a closed
    version stays one click away now that no menu lists them.
    """
    cite = ", ".join(
        part
        for part in (f"PRIMO leaderboard v{version.number}", updated_on(df), SPACE_URL)
        if part
    )
    items = []
    for listed in VERSIONS:
        state = "current" if listed == CURRENT else "closed"
        span = (
            f"since {listed.opened}"
            if listed == CURRENT
            else f"{listed.opened} to {_closed_on(listed)}"
        )
        link = (
            ""
            if listed == version
            else f'<a target="_self" href="{_href("tab=leaderboard", listed)}">'
            "View this board</a>"
        )
        items.append(
            f'<li class="{"pm-on" if listed == version else ""}">'
            f"<b>v{escape(listed.number)}</b><div>"
            f'<p class="pm-ver-meta"><span class="pm-chip pm-chip--{state}">{state}'
            f"</span> {escape(span)} · release {escape(listed.release)}</p>"
            f"<p>{escape(listed.notes)}</p></div>{link}</li>"
        )
    return (
        '<section class="pm-history" id="pm-history" '
        'aria-labelledby="pm-history-title"><div>'
        '<h2 class="pm-title" id="pm-history-title">Release notes</h2>'
        '<p class="pm-history-sub">What changed in each board version. A closed '
        "version keeps its board, read-only. To cite a rank: "
        f"<span>{escape(cite)}</span>.</p></div><ol>{''.join(items)}</ol></section>"
    )


def cta_html(boards: list[Board], version: Version = CURRENT) -> str:
    """The closing panel: an open call for cohorts, with examples of what is missing."""
    gaps = [b.name for group in SECTIONS for b in open_in_group(boards, group)]
    missing = ""
    if gaps:
        listed = gaps[0] if len(gaps) == 1 else f"{', '.join(gaps[:-1])} and {gaps[-1]}"
        missing = f"Still missing, for example: {listed}. "
    return (
        '<section class="pm-ask" aria-labelledby="pm-ask-title"><div>'
        '<p class="pm-label">Help us grow PRIMO</p>'
        '<h2 id="pm-ask-title">Which cohort should PRIMO cover next?</h2></div>'
        '<div class="pm-ask-go"><p>PRIMO is open to any omics layer and any disease '
        f"area. {escape(missing)}Bring a cohort, or propose a task on the cohorts "
        "already here.</p>"
        f'<a class="pm-btn" target="_self" href="{_href("tab=contribute", version)}">'
        "Contribute</a>"
        "</div></section>"
    )
