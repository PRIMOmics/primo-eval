"""Gradio front-end for the PRIMO public benchmark.

A left rail navigates six pages: Boards (a grid of every board), Board (one
leaderboard at a time), Tasks ("what is actually being tested?"), Submit ("how
do I enter?"), Contribute ("what is missing, and how do I add it?"), Method
("can I trust this?"). Every page is addressable -- ``?board=<slug>`` opens a
board and ``?tab=contribute`` opens a tab -- which is what the rail links and the
open cards use. The tab strip is hidden in CSS; the rail is the navigation.
``&version=<number>`` shows a closed board version, read-only: its registry at
its pinned commits and the rows it scored. Submit always shows the current one.

Choose one modality and upload its embedding file (rows keyed by ``dataset_id``
+ ``sample_id``); a fixed task probe scores each task (a dataset may carry
several hidden targets). Results roll up into boards -- the whole modality, one
therapeutic area, one task family -- and each board ranks the models that
covered every task our baseline scored, one column per category in its native
metric (AUROC, Pearson, or centered Spearman), plus a ``Mean`` of those columns
that orders the rows and is labelled as the cross-metric average it is.

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
page fetching for itself. It renders in two steps: every page but Boards first,
then Boards, which fits an Elo per board and would otherwise delay every page.
"""

import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import gradio as gr
import pandas as pd
from boards import PUBLIC, Board, build_boards, by_slug
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
    VERSION_PARAM,
    evaluation_confirmation,
    evaluation_status,
    rail_html,
    render_board,
    render_boards,
    render_tasks,
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
from versions import CURRENT, VERSIONS, Version

HERE = Path(__file__).parent
PAGES_DIR = HERE / "pages"
CSS_PATH = str(HERE / "primo.css")
ALLOWED_PATHS = [str(HERE / "assets"), str(HERE / "fonts")]
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
TAB_IDS = ("boards", "board", "tasks", "submit", "contribute", "method")
FOOT_TABS = frozenset({"tasks", "submit", "contribute", "method"})

THEME = gr.themes.Base(
    font=["Funnel Sans", "sans-serif"], font_mono=["DM Mono", "monospace"]
)

SUBMIT_HEAD = (
    '<div class="pm-head"><div><h1>Submit a model</h1>'
    "<p>One model, one modality. Partial coverage is fine within that modality. "
    "You are ranked on every board you cover in full.</p></div></div>"
)

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
    header.setAttribute("aria-sort", direction === 1 ? "ascending" : "descending");
  };

  document.addEventListener("click", (event) => {
    const header = event.target.closest("th.pm-sort");
    if (header) sort(header);
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


def _summary(result: dict, model_name: str) -> str:
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
        if result["missing"]:
            lines.append(f"- missing from file: {result['missing']}")
        if result["incomplete"]:
            lines.append(f"- could not score: {result['incomplete']}")
    return "\n".join(lines)


def _results_frame() -> pd.DataFrame:
    """Every published row, or an empty frame if the fetch fails.

    A Hugging Face hiccup must not block a submission, so an empty frame leaves
    the name free rather than raising: fail-open on the ownership check.
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


def _rendered_board(slug: str | None) -> str:
    """The Board page for ``slug`` -- re-rendered after a submission saves."""
    by_id, boards, df, _ = _page_state()
    return render_board(by_slug(boards, slug), df, by_id)


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
    slug: str | None,
    profile: gr.OAuthProfile | None,
):
    def _refuse(message: str):
        return message, _rendered_board(slug)

    if profile is None:
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
    if claimed and claimed != profile.username:
        return _refuse(
            f"The model name `{model}` is already submitted by another HF account. "
            "Resubmission of the same model is allowed only from the same HF account. "
            "Please pick another name."
        )
    try:
        result = score_all(
            submission_path, TOKEN, modality=modality, representation=EMBEDDING
        )
    except SubmissionError as error:
        return _refuse(f"❌ {error}")
    except EvaluatorError as error:
        traceback.print_exc()
        return _refuse(
            "⚠️ We couldn't evaluate your submission. This is on our side, not your "
            f"file. Please try again in a moment.\n\n`{error}`"
        )
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        return _refuse(f"⚠️ Unexpected evaluation error (our side): {error}")

    summary = _summary(result, model)
    now = datetime.now(timezone.utc)
    submitted_at = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    submission_id = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex}"
    credits = {
        OWNER: profile.username,
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
        return summary, _rendered_board(slug)

    hvg_result = _score_with_hvg(submission_path, modality)
    hvg_rows = (
        []
        if hvg_result is None
        else task_rows(
            hvg_name(model), hvg_result, submitted_at, submission_id, credits
        )
    )
    if hvg_result is not None:
        summary += "\n\n" + _summary(hvg_result, hvg_name(model))
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
        return summary, _rendered_board(slug)
    try:
        append_results(rows, TOKEN)
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        summary += f"\n\n⚠️ archived, but the leaderboard was not saved: {error}"
        return summary, _rendered_board(slug)

    summary += f"\n\n✅ **{model} is now shown on the relevant boards.**"
    if hvg_rows:
        summary += f" So is **{hvg_name(model)}**."
    try:
        append_submission(meta, TOKEN)
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        summary += f"\n\n⚠️ Contact metadata was not saved: {error}"
    return summary, _rendered_board(slug)


def _landing_tab(params: dict, has_board: bool) -> str:
    """Which page a visitor lands on: ``?tab=`` wins, then ``?board=``, else Boards.

    An unknown ``?tab=`` falls through to Boards rather than selecting nothing,
    which would render the Space with every panel collapsed.
    """
    tab = params.get("tab")
    if tab in TAB_IDS:
        return tab
    return "board" if has_board else "boards"


def _viewed_version(params: dict) -> Version:
    """The version a visitor asked for; Submit and unknown numbers get the current."""
    if params.get("tab") == "submit":
        return CURRENT
    number = params.get(VERSION_PARAM)
    return next((listed for listed in VERSIONS if listed.number == number), CURRENT)


def _init(request: gr.Request):
    """Render every page but Boards from one fetch, landing where the query asks.

    Navigation reloads the page, so the Board page is rendered only when a visitor
    lands on it. The fetched state is returned too, so ``_render_overview`` builds
    Boards from it without a second round trip.
    """
    params = dict(request.query_params) if request else {}
    state = _page_state(_viewed_version(params))
    by_id, boards, df, version = state
    board = by_slug(boards, params.get("board"))
    selected = _landing_tab(params, bool(params.get("board") and board))
    active_slug = board.slug if selected == "board" and board else None
    active_tab = selected if selected in FOOT_TABS else None
    return (
        gr.Tabs(selected=selected),
        rail_html(boards, active_slug, active_tab, version),
        board.slug if board else None,
        state,
        render_tasks(by_id),
        _about_text(by_id),
        render_board(board, df, by_id) if selected == "board" else "",
    )


def _render_overview(state: PageState) -> tuple[str, None]:
    """The Boards page, rendered after ``_init``.

    It rates every board, so it is the slowest page and must not hold back the
    one a visitor landed on. The state is then released, as nothing reads it again.
    """
    by_id, boards, df, version = state
    return render_boards(boards, df, by_id, version), None


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="PRIMO Benchmark",
        theme=THEME,
        head=PAGE_HEAD,
        css_paths=[CSS_PATH],
        js=TABLE_SORT_JS,
        fill_width=True,
    ) as demo:
        active_board = gr.State(None)
        page_state = gr.State(None)
        with gr.Row(elem_id="pm-shell"):
            with gr.Column(elem_id="pm-rail"):
                rail = gr.HTML()
            with gr.Column(elem_id="pm-main"):
                with gr.Tabs(elem_id="pm-pages") as pages:
                    with gr.Tab("Boards", id="boards"):
                        boards_html = gr.HTML()
                    with gr.Tab("Board", id="board"):
                        board_html = gr.HTML()
                    with gr.Tab("Tasks", id="tasks"):
                        tasks_html = gr.HTML()
                    with gr.Tab("Submit", id="submit"):
                        gr.HTML(SUBMIT_HEAD)
                        with gr.Column(elem_classes=["pm-body"]):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    gr.Markdown(_page_text("submit"))
                                with gr.Column(scale=2, elem_id="pm-form"):
                                    gr.LoginButton()
                                    modality_in = gr.Radio(
                                        MODALITY_CHOICES,
                                        label="Modality",
                                        info="One model submission covers one modality.",
                                    )
                                    download_md = gr.Markdown(_download_help(None))
                                    model_tb = gr.Textbox(
                                        label="Model name",
                                        placeholder="e.g. eva-rna-v1",
                                        info="Shown on the leaderboard.",
                                    )
                                    institution_tb = gr.Textbox(
                                        label="Institution (optional)",
                                        placeholder="e.g. Scienta",
                                        info="Shown on the leaderboard.",
                                    )
                                    author_submission_cb = gr.Checkbox(
                                        label="Submitted by the model's authors",
                                        info="Adds an Authors check next to the model name.",
                                    )
                                    email_tb = gr.Textbox(
                                        label="Email address",
                                        placeholder="you@lab.org",
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
                        with gr.Column(elem_classes=["pm-body", "pm-prose"]):
                            gr.Markdown(_page_text("contribute"))
                    with gr.Tab("Method", id="method"):
                        with gr.Column(elem_classes=["pm-body", "pm-prose"]):
                            about_md = gr.Markdown()

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
                active_board,
            ],
            [result_md, board_html],
        )
        evaluation_event.then(
            lambda: gr.update(visible=False),
            None,
            status_html,
            show_progress="hidden",
        )
        modality_in.change(_download_help, modality_in, download_md)
        demo.load(
            _init,
            None,
            [pages, rail, active_board, page_state, tasks_html, about_md, board_html],
        ).success(_render_overview, page_state, [boards_html, page_state])
    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        allowed_paths=ALLOWED_PATHS,
    )
