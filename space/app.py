"""Gradio front-end for the PRIMO public benchmark.

A top bar navigates five pages: Leaderboard (one ranking, chosen with the
Modality, Therapeutic area and Task pills), Tasks ("what is actually being
tested?"), Submit ("how do I enter?"), Contribute ("what is missing, and how do
I add it?"), Method ("can I trust this?"). Every page is addressable --
``?tab=contribute`` opens a tab and ``?modality=bulk-rna&area=oncology&task=all``
a view; old ``?board=<slug>`` links still open their view. The tab strip is
hidden in CSS; the top bar is the navigation.
``&version=<number>`` shows a closed board version, read-only: its registry at
its pinned commits and the rows it scored. Submit always shows the current one.

Choose one modality and upload its embedding file (rows keyed by ``dataset_id``
+ ``sample_id``); a fixed task probe scores each task (a dataset may carry
several hidden targets). Results roll up into boards -- the whole modality, one
therapeutic area, one task family -- and each board ranks the models that
covered every task our baseline scored, one column per category in its native
metric (AUROC, Pearson, or centered Spearman). The rows are ordered by Elo; no
column averages across metrics.

Disclosure policy -- what the public pages may show:
  per task   disease, tissue, area, modality, what is predicted, class names, n, metric
  aggregated the public archives the cohorts sit in, and their licences
  never      study accessions, dataset_id -> cohort, hub keys, per-sample labels,
             pairing identifiers or post-treatment responses
The public tasks are the exception: their labels and responses are published in
the public data repo, and the pages tag them ``Public``.
Naming the accession behind ``d002`` would put every label one GEO download
away, so ``sources`` is collapsed to its archive (NCBI GEO / EMBL-EBI
ArrayExpress) and ``citation`` is never rendered at all. Submitters can publish
an optional institution, paper link and an author-submission check; legacy private
usernames remain hidden.

Failures are surfaced by who owns them: a bad file tells the submitter exactly
what to fix; an evaluator-side failure says "our side, please retry" and logs
the traceback rather than blaming the submission.

A page load is ONE fetch: ``_init`` pulls the registry, the boards and the
persisted results once and threads that state into every page, rather than each
page fetching for itself. The state stays in the session, so a click on a pill
re-renders the view from it, with no reload and no new fetch.
"""

import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import gradio as gr
import pandas as pd
from boards import (
    ALL,
    PUBLIC,
    Board,
    area_choices,
    board_view,
    build_boards,
    category_choices,
    modality_choices,
    view_board,
    without_public,
)
from evaluator import (
    EMBEDDING,
    EvaluatorError,
    SubmissionError,
    _norm_id,
    fetch_manifest,
    fetch_public_task_ids,
    fetch_tasks_registry,
    manifest_ids,
    score_all,
    scoreable_tasks,
)
from leaderboard import RESERVED_COLUMNS, source_repositories
from render import (
    CREDIT,
    VERSION_PARAM,
    cta_html,
    evaluation_confirmation,
    evaluation_status,
    hero_html,
    history_html,
    nav_html,
    render_tasks,
    render_view,
)
from results import (
    BASELINE_TAG,
    HVG_TAG,
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    OWNER,
    PAPER_LINK,
    RESULT_COLUMNS,
    SUBMISSION_ID,
    append_results,
    append_submission,
    archive_submission,
    fetch_hvg_reference,
    hvg_name,
    owner_of,
    read_results,
    task_rows,
    version_rows,
)
from submit_api import (
    ERROR,
    MAX_FILE_SIZE,
    PUBLISHED,
    REFUSED,
    SUBMISSIONS,
    SUBMISSIONS_PER_DAY,
    SUBMITTED_AT,
    recent_submissions,
    reply,
    route,
)
from versions import CURRENT, VERSIONS, Version

HERE = Path(__file__).parent
PAGES_DIR = HERE / "pages"
CSS_PATH = str(HERE / "primo.css")
ALLOWED_PATHS = [str(HERE / "assets")]
SOCIAL_PREVIEW_URL = (
    "https://huggingface.co/spaces/PRIMOmics/primo-eval/resolve/main/"
    "assets/primo-social-preview.png"
)
PAGE_HEAD = f"""
<meta property="og:title" content="PRIMO Benchmark" />
<meta property="og:description" content="A blind benchmark for omics foundation models." />
<meta property="og:image" content="{SOCIAL_PREVIEW_URL}" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="PRIMO Benchmark" />
<meta name="twitter:description" content="A blind benchmark for omics foundation models." />
<meta name="twitter:image" content="{SOCIAL_PREVIEW_URL}" />
"""

TOKEN = os.environ.get("HF_TOKEN")
TAB_IDS = ("leaderboard", "tasks", "submit", "contribute", "method")
LEGACY_TABS = {"boards": "leaderboard", "board": "leaderboard"}
DEFAULT_MODALITY = "bulk-rna"
SHOW = "show"
HIDE = "hide"
TOGGLE_CHOICES = [("Show", SHOW), ("Hide", HIDE)]

THEME = gr.themes.Base(
    font=[gr.themes.LocalFont("IBM Plex Sans", weights=(400, 700)), "sans-serif"],
    font_mono=[gr.themes.LocalFont("IBM Plex Mono", weights=(400, 700)), "monospace"],
)

VIEW_URL_JS = """
(modality, area, task, hvg, pub) => {
  const url = new URL(window.location.href);
  const values = { modality, area, task, hvg, public: pub };
  const defaults = {
    modality: "bulk-rna", area: "all", task: "all", hvg: "show", public: "show",
  };
  for (const [key, value] of Object.entries(values)) {
    if (!value || value === defaults[key]) url.searchParams.delete(key);
    else url.searchParams.set(key, value);
  }
  url.searchParams.delete("board");
  window.history.replaceState(null, "", url);
}
"""

SUBMIT_HEAD = (
    '<div class="pm-head"><div><h1>Submit a model</h1>'
    "<p>One model, one modality. Partial coverage is fine within that modality. "
    "You are ranked on every board you cover in full.</p></div></div>"
)
PAGE_HEADS = {
    "contribute": (
        '<div class="pm-head"><div><h1>Contribute</h1>'
        "<p>What PRIMO is missing, and how to add it.</p></div></div>"
    ),
    "method": (
        '<div class="pm-head"><div><h1>Method</h1>'
        "<p>How a model is scored, and what the pages show.</p></div></div>"
    ),
}

MODALITY_CHOICES = [
    ("Bulk RNAseq", "bulk-rna"),
    ("Single-cell RNAseq", "single-cell-rna"),
]

TABLE_SORT_JS = """
() => {
  if (window.pmTableSortBound) return;
  window.pmTableSortBound = true;

  const value = (cell) => {
    const text = cell.dataset.sortValue || cell.textContent.trim();
    if (!text || text.toLowerCase() === "n/a") return { missing: true, text };
    const number = Number(text.replaceAll(",", ""));
    return Number.isFinite(number) ? { missing: false, number, text } : { missing: false, text };
  };

  const sort = (header) => {
    const table = header.closest("table.pm-table");
    const body = table?.tBodies[0];
    if (!body) return;
    const index = Number(header.dataset.sortIndex);
    const direction = header.dataset.sortDirection === "asc" ? -1 : 1;
    const rows = Array.from(body.rows);

    rows.sort((left, right) => {
      const a = value(left.cells[index]);
      const b = value(right.cells[index]);
      if (a.missing || b.missing) return a.missing === b.missing ? 0 : a.missing ? 1 : -1;
      if (a.number !== undefined && b.number !== undefined) return direction * (a.number - b.number);
      return direction * a.text.localeCompare(b.text, undefined, { numeric: true });
    });
    rows.forEach((row) => body.append(row));
    table.querySelectorAll("th.pm-sort").forEach((cell) => {
      cell.dataset.sortDirection = "";
      cell.setAttribute("aria-sort", "none");
    });
    header.dataset.sortDirection = direction === 1 ? "asc" : "desc";
    table.dataset.sorted = index === 0 && direction === 1 ? "" : "true";
    header.setAttribute("aria-sort", direction === 1 ? "ascending" : "descending");
  };

  document.addEventListener("click", (event) => {
    const header = event.target.closest("th.pm-sort");
    if (header) sort(header);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest("#pm-cite-btn")) return;
    document.querySelector("#pm-cite-dialog")?.showModal();
  });
  document.addEventListener("click", (event) => {
    const evaluateButton = event.target.closest(
      "#pm-evaluate button, button#pm-evaluate"
    );
    if (!evaluateButton) return;
    if (evaluateButton.dataset.confirmed === "true") {
      delete evaluateButton.dataset.confirmed;
      return;
    }

    event.preventDefault();
    event.stopImmediatePropagation();
    const dialog = document.querySelector("#pm-evaluation-confirmation");
    if (!dialog) return;
    dialog.returnValue = "";
    dialog.addEventListener("close", () => {
      if (dialog.returnValue !== "confirm") return;
      evaluateButton.dataset.confirmed = "true";
      evaluateButton.click();
    }, { once: true });
    dialog.showModal();
  }, true);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const header = event.target.closest("th.pm-sort");
    if (!header) return;
    event.preventDefault();
    sort(header);
  });
  document.addEventListener("input", (event) => {
    const box = event.target.closest(".pm-find");
    const table = box?.closest(".pm-more")?.querySelector("table.pm-table");
    if (!table) return;
    const query = box.value.trim().toLowerCase();
    table.querySelectorAll("th[data-score]").forEach((header) => {
      const hidden = !header.textContent.toLowerCase().includes(query);
      for (const row of table.rows) row.cells[header.cellIndex].hidden = hidden;
    });
    const wrap = table.parentElement;
    wrap.scrollLeft = query ? wrap.scrollWidth : 0;
  });
}
"""


def _page_text(name: str) -> str:
    """One ``pages/<tab>.md`` per prose tab, named after the tab it fills.

    Editing the site's words never means touching Python. The data pages (Boards,
    Board, Tasks) are generated markup instead.
    """
    return (PAGES_DIR / f"{name}.md").read_text()


def _download_help(modality: str | None) -> str:
    """Show the quickstart command for the selected modality."""
    if not modality:
        return "Select a modality to get its download command."
    return (
        f"```bash\npython quickstart.py --modality {modality} "
        "--out submission.parquet\n```"
    )


def _registry_by_id(version: Version) -> dict[str, dict]:
    """The scoreable tasks of ``version`` by task_id, flagged ``public`` if so.

    ``PRIMO_FIXTURE=1`` swaps the token-gated fetch for the test registry, so the
    app runs offline for local smoke-testing.
    """
    if os.environ.get("PRIMO_FIXTURE"):
        from conftest import REGISTRY

        return {task_id: dict(task) for task_id, task in REGISTRY.items()}
    datasets = manifest_ids(fetch_manifest(TOKEN, version.data))
    registry = scoreable_tasks(fetch_tasks_registry(TOKEN, version.labels), datasets)
    by_id = {_norm_id(task["task_id"]): task for task in registry}
    public = fetch_public_task_ids(TOKEN, version.data)
    return {
        task_id: {**task, PUBLIC: task_id in public} for task_id, task in by_id.items()
    }


PageState = tuple[dict[str, dict], list[Board], pd.DataFrame, Version]


def _page_state(version: Version = CURRENT) -> PageState:
    """Registry, boards and results of ``version`` -- one fetch per render.

    A failed fetch yields empty structures so the page renders a "come back
    later" state instead of a stack trace. Every page is built from one of these,
    threaded through rather than refetched, so a page load is one round trip.
    """
    try:
        by_id = _registry_by_id(version)
        rows = version_rows(read_results(TOKEN), version.number)
        return by_id, build_boards(by_id), rows, version
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return {}, [], pd.DataFrame(columns=RESULT_COLUMNS), version


def _about_text(by_id: dict[str, dict]) -> str:
    """Methodology + the archives the cohorts live in, never their accessions.

    The placeholder is substituted, not ``.format``-ed: a page of prose is free
    to contain a brace, and a stray one must not blow up the tab. An unreachable
    registry leaves ``by_id`` empty and the sentence falls back to a generic one.
    """
    repositories = source_repositories(list(by_id.values()))
    return _page_text("about").replace(
        "{repositories}", " and ".join(repositories) or "public archives"
    )


def _summary(result: dict, model_name: str, *, list_missing: bool = True) -> str:
    lines = [
        f"**{model_name}**: covered {result['n_datasets_scored']}/"
        f"{result['n_datasets_total']} datasets",
        "",
    ]
    for category, stats in sorted(result["categories"].items()):
        lines.append(f"- **{category}** ({stats['metric']}) = {stats['mean']:.3f}")
    if result["full_coverage"]:
        lines.append("\n✅ **full coverage.** You are ranked on every board.")
    else:
        lines.append(
            "\n⚠️ **partial coverage.** You are ranked on the boards whose scored "
            "tasks you covered in full, and your scores always appear in each "
            "board's **per-task** table."
        )
        if result["missing"] and list_missing:
            lines.append(f"- missing from file: {result['missing']}")
        if result["incomplete"]:
            lines.append(f"- could not score: {result['incomplete']}")
    return "\n".join(lines)


def _results_frame() -> pd.DataFrame:
    """Every published row, or an empty frame if the fetch fails.

    A Hugging Face hiccup must not block a submission, so an empty frame leaves
    the name free rather than raising: fail-open on the ownership check and
    on the limit of submissions per day.
    """
    try:
        return read_results(TOKEN)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return pd.DataFrame(columns=RESULT_COLUMNS)


def _claimed_by(df: pd.DataFrame, model: str) -> str:
    """Who already owns this model name, or ``""``."""
    return owner_of(df, model)


def _score_with_hvg(submission_path: str, modality: str) -> dict | None:
    """Score the submission concatenated with the HVG features, or None on failure.

    The plain scoring already validated the file, so a failure here is ours: it
    is logged and the plain entry is saved alone rather than blocked.
    """
    try:
        return score_all(
            submission_path,
            TOKEN,
            modality=modality,
            representation=EMBEDDING,
            reference=fetch_hvg_reference(TOKEN),
        )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return None


def run_submission(
    submission_path: str | None,
    modality: str | None,
    model_name: str,
    institution: str,
    is_author_submission: bool,
    email: str,
    paper_link: str,
    hf_model_link: str,
    notes: str,
    username: str | None,
) -> dict:
    """Check, score and save one submission, for the web form and ``/submit``.

    The reply holds the ``status``, ``message`` and ``submission_id`` that
    ``primo submit`` reads.
    """

    def _refuse(message: str) -> dict:
        return reply(REFUSED, message)

    if not username:
        return _refuse("Please sign in with Hugging Face to submit.")
    if not modality:
        return _refuse("Please select a modality.")
    if not submission_path:
        return _refuse("Please upload a submission file.")
    if not model_name or not model_name.strip():
        return _refuse("Please enter a model name.")
    if not email or not email.strip():
        return _refuse("Please enter a contact email.")
    model = model_name.strip()
    if BASELINE_TAG in model.lower():
        return _refuse(
            f"`{BASELINE_TAG}` is reserved for our reference submissions. Please "
            "pick another model name."
        )
    if HVG_TAG.lower() in model.lower():
        return _refuse(
            f"`{HVG_TAG}` is reserved for the entry we derive from every "
            "submission. Please pick another model name."
        )
    if model in RESERVED_COLUMNS:
        return _refuse(
            f"`{model}` is a column of the per-task table. Please pick another "
            "model name."
        )
    published = _results_frame()
    claimed = _claimed_by(published, model)
    if claimed and claimed != username:
        return _refuse(
            f"The model name `{model}` is already submitted by another HF account. "
            "Resubmission of the same model is allowed only from the same HF account. "
            "Please pick another name."
        )
    if (
        recent_submissions(published, username, datetime.now(timezone.utc))
        >= SUBMISSIONS_PER_DAY
    ):
        return _refuse(
            f"An account can publish {SUBMISSIONS_PER_DAY} submissions in 24 "
            "hours. Please try again later."
        )
    try:
        result = score_all(
            submission_path, TOKEN, modality=modality, representation=EMBEDDING
        )
    except SubmissionError as error:
        return _refuse(f"❌ {error}")
    except EvaluatorError as error:
        traceback.print_exc()
        return reply(
            ERROR,
            "⚠️ We couldn't evaluate your submission. This is on our side, not your "
            f"file. Please try again in a moment.\n\n`{error}`",
        )
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        return reply(ERROR, f"⚠️ Unexpected evaluation error (our side): {error}")

    summary = _summary(result, model)
    now = datetime.now(timezone.utc)
    submitted_at = now.strftime(SUBMITTED_AT)
    submission_id = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex}"
    credits = {
        OWNER: username,
        INSTITUTION: (institution or "").strip(),
        IS_AUTHOR_SUBMISSION: bool(is_author_submission),
        PAPER_LINK: (paper_link or "").strip(),
    }
    rows = task_rows(model, result, submitted_at, submission_id, credits)
    if not rows:
        summary += (
            "\n\n⚠️ No valid scores were produced, so the model was not added to "
            "any board."
        )
        return _refuse(summary)

    hvg_result = _score_with_hvg(submission_path, modality)
    hvg_rows = (
        []
        if hvg_result is None
        else task_rows(
            hvg_name(model), hvg_result, submitted_at, submission_id, credits
        )
    )
    if hvg_result is not None:
        summary += "\n\n" + _summary(hvg_result, hvg_name(model), list_missing=False)
    if not hvg_rows:
        summary += (
            f"\n\n⚠️ The `{hvg_name(model)}` entry could not be scored (our side), "
            "so only the plain entry is saved; an earlier "
            f"`{hvg_name(model)}` entry may still show older scores."
        )
    rows += hvg_rows

    meta = {
        "model_name": model,
        "submitted_at": submitted_at,
        SUBMISSION_ID: submission_id,
        "email": email.strip(),
        **credits,
        "hf_model_link": (hf_model_link or "").strip(),
        "notes": (notes or "").strip(),
    }
    try:
        archive_submission(
            submission_path, model, submission_id, modality, submitted_at, TOKEN
        )
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        summary += f"\n\n⚠️ scored, but the embedding archive was not saved: {error}"
        return reply(ERROR, summary)
    try:
        append_results(rows, TOKEN)
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        summary += f"\n\n⚠️ archived, but the leaderboard was not saved: {error}"
        return reply(ERROR, summary, submission_id)

    summary += f"\n\n✅ **{model} is now shown on the relevant boards.**"
    if hvg_rows:
        summary += f" So is **{hvg_name(model)}**."
    try:
        append_submission(meta, TOKEN)
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        summary += f"\n\n⚠️ Contact metadata was not saved: {error}"
    return reply(PUBLISHED, summary, submission_id)


def evaluate(
    submission_path: str,
    modality: str | None,
    model_name: str,
    institution: str,
    is_author_submission: bool,
    email: str,
    paper_link: str,
    hf_model_link: str,
    notes: str,
    profile: gr.OAuthProfile | None,
):
    """The web form: the shared submission path, then the page's answer."""
    return run_submission(
        submission_path,
        modality,
        model_name,
        institution,
        is_author_submission,
        email,
        paper_link,
        hf_model_link,
        notes,
        None if profile is None else profile.username,
    )["message"]


def _landing_tab(params: dict) -> str:
    """Which page a visitor lands on: ``?tab=``, old board links included.

    An unknown ``?tab=`` falls through to the leaderboard rather than selecting
    nothing, which would render the Space with every panel collapsed.
    """
    tab = params.get("tab")
    tab = LEGACY_TABS.get(tab, tab)
    return tab if tab in TAB_IDS else "leaderboard"


def _initial_view(
    params: dict, by_id: dict[str, dict], boards: list[Board]
) -> tuple[str, str, str]:
    """The pills a visitor lands on: an old ``?board=``, else the query, else Bulk."""
    old = next((b for b in boards if b.slug == params.get("board")), None)
    if old is not None:
        return board_view(old, by_id)
    modalities = [slug for _, slug in modality_choices(by_id)]
    modality = params.get("modality")
    if modality not in modalities:
        modality = (
            DEFAULT_MODALITY
            if DEFAULT_MODALITY in modalities or not modalities
            else modalities[0]
        )
    areas = [slug for _, slug in area_choices(by_id, modality)]
    categories = [slug for _, slug in category_choices(by_id, modality)]
    area = params.get("area") if params.get("area") in areas else ALL
    category = params.get("task") if params.get("task") in categories else ALL
    return modality, area, category


def _view_name(by_id: dict[str, dict], modality: str, area: str, category: str) -> str:
    """The view as its pills read, ``Bulk RNAseq · Oncology · All tasks``."""
    modalities = {slug: name for name, slug in modality_choices(by_id)}
    areas = {slug: name for name, slug in area_choices(by_id, modality)}
    categories = {slug: name for name, slug in category_choices(by_id, modality)}
    return " · ".join(
        (
            modalities.get(modality, modality),
            "All areas" if area == ALL else areas.get(area, area),
            "All tasks" if category == ALL else categories.get(category, category),
        )
    )


def _render_view(
    state: PageState, modality: str, area: str, category: str, hvg: str, public: str
) -> str:
    """The tables of the view the pills select, on the version on screen.

    A pill clicked before the page has loaded has no state yet, and changes nothing.
    """
    if state is None:
        return gr.skip()
    by_id, _, df, _ = state
    board = view_board(by_id, modality, area, category)
    if public == HIDE:
        board = without_public(board, by_id)
    name = _view_name(by_id, modality, area, category)
    return render_view(board, df, by_id, name, hvg != HIDE)


def _on_modality(state: PageState, modality: str, hvg: str, public: str):
    """A new modality brings its own area and task pills, both back on All."""
    by_id = state[0]
    return (
        gr.Radio(choices=area_choices(by_id, modality), value=ALL),
        gr.Radio(choices=category_choices(by_id, modality), value=ALL),
        _render_view(state, modality, ALL, ALL, hvg, public),
    )


def _viewed_version(params: dict) -> Version:
    """The version a visitor asked for; Submit and unknown numbers get the current."""
    if params.get("tab") == "submit":
        return CURRENT
    number = params.get(VERSION_PARAM)
    return next((listed for listed in VERSIONS if listed.number == number), CURRENT)


def _init(request: gr.Request):
    """Render every page from one fetch, on the tab and the view the query asks.

    The fetched state is returned too, so a click on a pill re-renders the view
    without a second round trip.
    """
    params = dict(request.query_params) if request else {}
    state = _page_state(_viewed_version(params))
    by_id, boards, df, version = state
    tab = _landing_tab(params)
    modality, area, category = _initial_view(params, by_id, boards)
    hvg = HIDE if params.get("hvg") == HIDE else SHOW
    public = HIDE if params.get("public") == HIDE else SHOW
    return (
        gr.Tabs(selected=tab),
        nav_html(tab, version),
        hero_html(boards, df, version),
        _render_view(state, modality, area, category, hvg, public),
        gr.Radio(choices=modality_choices(by_id), value=modality),
        gr.Radio(choices=area_choices(by_id, modality), value=area),
        gr.Radio(choices=category_choices(by_id, modality), value=category),
        gr.Radio(value=hvg),
        gr.Radio(value=public, visible=any(t.get(PUBLIC) for t in by_id.values())),
        state,
        render_tasks(by_id),
        _about_text(by_id),
        history_html(df, version),
        cta_html(boards, version),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="PRIMO Benchmark", fill_width=True) as demo:
        page_state = gr.State(None)
        with gr.Column(elem_id="pm-site"):
            nav = gr.HTML(padding=False)
            with gr.Tabs(elem_id="pm-pages") as pages:
                with gr.Tab("Leaderboard", id="leaderboard"):
                    hero = gr.HTML(padding=False)
                    with gr.Row(elem_id="pm-controls"):
                        modality_f = gr.Radio(
                            label="Modality", elem_classes=["pm-seg", "pm-seg--main"]
                        )
                        area_f = gr.Radio(
                            label="Therapeutic area",
                            elem_classes=["pm-seg", "pm-seg--line"],
                        )
                        task_f = gr.Radio(
                            label="Task", elem_classes=["pm-seg", "pm-seg--line"]
                        )
                        hvg_f = gr.Radio(
                            TOGGLE_CHOICES,
                            value=SHOW,
                            label="+ HVG models",
                            elem_classes=["pm-seg", "pm-seg--minor"],
                        )
                        public_f = gr.Radio(
                            TOGGLE_CHOICES,
                            value=SHOW,
                            label="Public tasks",
                            visible=False,
                            elem_classes=["pm-seg", "pm-seg--minor"],
                        )
                    board_html = gr.HTML(padding=False)
                    history = gr.HTML(padding=False)
                    cta = gr.HTML(padding=False)
                with gr.Tab("Tasks", id="tasks"):
                    tasks_html = gr.HTML(padding=False)
                with gr.Tab("Submit", id="submit"):
                    gr.HTML(SUBMIT_HEAD, padding=False)
                    with gr.Column(elem_classes=["pm-body"]):
                        with gr.Row(elem_id="pm-submit"):
                            with gr.Column(scale=3):
                                gr.Markdown(
                                    _page_text("submit"),
                                    elem_classes=["pm-prose", "pm-submit-text"],
                                )
                            with gr.Column(scale=2, elem_id="pm-form"):
                                gr.LoginButton()
                                modality_in = gr.Radio(
                                    MODALITY_CHOICES,
                                    label="Modality",
                                    info="One model submission covers one modality.",
                                    elem_classes=["pm-seg", "pm-seg--form"],
                                )
                                download_md = gr.Markdown(
                                    _download_help(None), elem_classes=["pm-hint"]
                                )
                                model_tb = gr.Textbox(
                                    label="Model name",
                                    placeholder="e.g. my-model-v1",
                                    info="Shown on the leaderboard.",
                                )
                                institution_tb = gr.Textbox(
                                    label="Institution (optional)",
                                    placeholder="e.g. Example University",
                                    info="Shown on the leaderboard.",
                                )
                                author_submission_cb = gr.Checkbox(
                                    label="Submitted by the model's authors",
                                    info="Adds an Authors check next to the model name.",
                                )
                                email_tb = gr.Textbox(
                                    label="Email address",
                                    placeholder="Your contact email",
                                    info="Contact for this submission, kept private.",
                                )
                                notes_tb = gr.Textbox(
                                    label="Training data / notes (optional)",
                                    placeholder="e.g. pretrained on atlas X",
                                    info="About the model or its training data.",
                                )
                                paper_tb = gr.Textbox(
                                    label="Paper link (optional)",
                                    placeholder="https://arxiv.org/abs/...",
                                    info="Links the model name on the leaderboard.",
                                )
                                hf_tb = gr.Textbox(
                                    label="Hugging Face model link (optional)",
                                    placeholder="https://huggingface.co/...",
                                )
                                file_in = gr.File(
                                    label="Submission (.csv / .tsv / .parquet / .npz)",
                                    type="filepath",
                                )
                                run_btn = gr.Button(
                                    "Evaluate",
                                    elem_id="pm-evaluate",
                                    elem_classes=[
                                        "pm-btn",
                                        "pm-btn--primary",
                                        "pm-btn--block",
                                    ],
                                )
                                status_html = gr.HTML(
                                    evaluation_status(), visible=False
                                )
                                gr.HTML(evaluation_confirmation())
                        result_md = gr.Markdown()
                with gr.Tab("Contribute", id="contribute"):
                    gr.HTML(PAGE_HEADS["contribute"], padding=False)
                    with gr.Column(elem_classes=["pm-body", "pm-prose"]):
                        gr.Markdown(_page_text("contribute"))
                with gr.Tab("Method", id="method"):
                    gr.HTML(PAGE_HEADS["method"], padding=False)
                    with gr.Column(elem_classes=["pm-body", "pm-prose"]):
                        about_md = gr.Markdown()
            gr.HTML(CREDIT, padding=False)

        evaluation_event = run_btn.click(
            lambda: gr.update(visible=True),
            None,
            status_html,
            show_progress="hidden",
        ).then(
            evaluate,
            [
                file_in,
                modality_in,
                model_tb,
                institution_tb,
                author_submission_cb,
                email_tb,
                paper_tb,
                hf_tb,
                notes_tb,
            ],
            result_md,
            concurrency_id=SUBMISSIONS,
        )
        gr.api(route(run_submission), api_name="submit", concurrency_id=SUBMISSIONS)
        evaluation_event.then(
            lambda: gr.update(visible=False),
            None,
            status_html,
            show_progress="hidden",
        )
        modality_in.change(_download_help, modality_in, download_md)

        pills = [modality_f, area_f, task_f, hvg_f, public_f]
        modality_f.input(
            _on_modality,
            [page_state, modality_f, hvg_f, public_f],
            [area_f, task_f, board_html],
            show_progress="hidden",
        ).then(None, pills, None, js=VIEW_URL_JS)
        for pill in pills[1:]:
            pill.input(
                _render_view,
                [page_state, *pills],
                board_html,
                show_progress="hidden",
            ).then(None, pills, None, js=VIEW_URL_JS)
        demo.load(
            _init,
            None,
            [
                pages,
                nav,
                hero,
                board_html,
                *pills,
                page_state,
                tasks_html,
                about_md,
                history,
                cta,
            ],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        allowed_paths=ALLOWED_PATHS,
        max_file_size=MAX_FILE_SIZE,
        theme=THEME,
        head=PAGE_HEAD,
        css_paths=[CSS_PATH],
        js=TABLE_SORT_JS,
    )
