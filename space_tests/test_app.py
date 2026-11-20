"""Unit tests for submission feedback in the PRIMO app."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from boards import ALL, build_boards
from versions import CURRENT

from .conftest import CLOSED, REGISTRY

previous_gradio = sys.modules.get("gradio")
gradio_stub = ModuleType("gradio")
gradio_stub.themes = SimpleNamespace(
    Base=lambda **kwargs: kwargs, LocalFont=lambda name, weights: (name, weights)
)
gradio_stub.OAuthProfile = object
gradio_stub.Request = object
gradio_stub.Blocks = object
gradio_stub.Tabs = lambda selected: selected
gradio_stub.Radio = lambda **kwargs: kwargs
gradio_stub.skip = lambda: "skip"
sys.modules["gradio"] = gradio_stub
app = importlib.import_module("app")
if previous_gradio is None:
    sys.modules.pop("gradio")
else:
    sys.modules["gradio"] = previous_gradio


def _ledger(*rows) -> pd.DataFrame:
    """A ``task_results.csv`` frame of ``(model_name, hf_username, task_id)`` rows."""
    return pd.DataFrame(
        [
            {
                "model_name": model,
                "hf_username": owner,
                "task_id": task_id,
                "score": 0.5,
                "submitted_at": "2026-01-01",
                "is_baseline": False,
            }
            for model, owner, task_id in rows
        ],
        columns=[
            "model_name",
            "hf_username",
            "task_id",
            "score",
            "submitted_at",
            "is_baseline",
        ],
    )


def _result() -> dict:
    return {
        "n_datasets_scored": 1,
        "n_datasets_total": 1,
        "categories": {"clinical_scores": {"metric": "pearson", "mean": 0.75}},
        "full_coverage": True,
        "missing": [],
        "incomplete": [],
        "per_task": [
            SimpleNamespace(
                task_id="t901", score=0.75, repeat_scores=[], diagnostics={}
            )
        ],
    }


def _evaluate(
    monkeypatch,
    append_results,
    result=None,
    published=None,
    model="eva-rna-v1",
    hvg_fails=False,
) -> str:
    def score_all(path, token, modality, representation, reference=None):
        if reference is not None and hvg_fails:
            raise RuntimeError("reference unavailable")
        return result or _result()

    monkeypatch.setattr(app, "_claimed_by", lambda df, model: "")
    monkeypatch.setattr(
        app, "_results_frame", lambda: published if published is not None else _ledger()
    )
    monkeypatch.setattr(app, "score_all", score_all)
    monkeypatch.setattr(app, "fetch_hvg_reference", lambda token: "hvg.parquet")
    monkeypatch.setattr(app, "append_results", append_results)
    monkeypatch.setattr(app, "archive_submission", lambda *args: {})
    monkeypatch.setattr(app, "append_submission", lambda meta, token: None)

    return app.evaluate(
        "submission.csv",
        "bulk-rna",
        model,
        "Scienta",
        True,
        "person@example.org",
        "",
        "",
        "",
        SimpleNamespace(username="submitter"),
    )


def test_successful_submission_confirms_model_is_on_relevant_boards(monkeypatch):
    summary = _evaluate(monkeypatch, lambda rows, token: None)

    assert "eva-rna-v1 is now shown on the relevant boards" in summary


def test_table_sort_prefers_the_numeric_cell_value_over_sd_display_text():
    assert "cell.dataset.sortValue || cell.textContent.trim()" in app.TABLE_SORT_JS


def test_a_column_sort_marks_the_table_so_the_top_three_frame_hides():
    assert "table.dataset.sorted" in app.TABLE_SORT_JS


def test_the_cite_button_opens_its_window():
    assert 'querySelector("#pm-cite-dialog")?.showModal()' in app.TABLE_SORT_JS


def test_the_find_box_hides_the_score_columns_of_other_models():
    assert 'event.target.closest(".pm-find")' in app.TABLE_SORT_JS
    assert 'table.querySelectorAll("th[data-score]")' in app.TABLE_SORT_JS


def test_a_top_bar_link_switches_the_page_without_a_reload():
    assert '#pm-pages button[role="tab"]' in app.TABLE_SORT_JS
    assert "window.history.pushState" in app.TABLE_SORT_JS
    assert '"popstate"' in app.TABLE_SORT_JS


def test_a_link_to_another_board_version_still_reloads():
    assert (
        'target.searchParams.get("version") !== url.searchParams.get("version")'
        in app.TABLE_SORT_JS
    )


def test_owned_model_name_does_not_reveal_owner(monkeypatch):
    monkeypatch.setattr(app, "_results_frame", lambda: _ledger())
    monkeypatch.setattr(app, "_claimed_by", lambda df, model: "private-owner")

    summary = app.evaluate(
        "submission.csv",
        "bulk-rna",
        "eva-rna-v1",
        "Scienta",
        True,
        "person@example.org",
        "",
        "",
        "",
        SimpleNamespace(username="submitter"),
    )

    assert summary == (
        "The model name `eva-rna-v1` is already submitted by another HF account. "
        "Resubmission of the same model is allowed only from the same HF account. "
        "Please pick another name."
    )
    assert "private-owner" not in summary


def test_failed_leaderboard_save_does_not_claim_model_is_visible(monkeypatch):
    def fail(rows, token):
        raise RuntimeError("Hub unavailable")

    summary = _evaluate(monkeypatch, fail)

    assert "is now shown on the relevant boards" not in summary
    assert "leaderboard was not saved" in summary


def test_submission_without_valid_scores_is_not_claimed_or_persisted(monkeypatch):
    persisted = []
    result = _result()
    result["per_task"] = []

    summary = _evaluate(
        monkeypatch,
        lambda rows, token: persisted.extend(rows),
        result,
    )

    assert persisted == []
    assert "is now shown on the relevant boards" not in summary
    assert "No valid scores were produced" in summary


def test_registry_flags_the_public_tasks(monkeypatch):
    monkeypatch.delenv("PRIMO_FIXTURE", raising=False)
    monkeypatch.setattr(app, "fetch_manifest", lambda token, revision: [{"id": "d901"}])
    monkeypatch.setattr(
        app,
        "fetch_tasks_registry",
        lambda token, revision: [
            {"task_id": "t901", "dataset_id": "d901"},
            {"task_id": "t902", "dataset_id": "d901"},
        ],
    )
    monkeypatch.setattr(app, "fetch_public_task_ids", lambda token, revision: {"t902"})
    registry = app._registry_by_id(CURRENT)
    assert registry["t901"]["public"] is False
    assert registry["t902"]["public"] is True


def _init(monkeypatch, registry, boards, **query):
    empty = pd.DataFrame(columns=app.RESULT_COLUMNS)
    monkeypatch.setattr(
        app,
        "_page_state",
        lambda version, revision=None: (registry, boards, empty, version),
    )
    monkeypatch.setattr(app, "results_hash", lambda token: ("hash-1", "commit-1"))
    monkeypatch.setattr(app, "PAGES_BY_VERSION", {})
    monkeypatch.setattr(app.gr, "Tabs", lambda **kwargs: kwargs, raising=False)
    return app._init(SimpleNamespace(query_params=query))


def test_every_page_lands_on_its_tab_and_renders_the_view(
    monkeypatch, registry, boards
):
    on_tasks = _init(monkeypatch, registry, boards, tab="tasks")
    on_boards = _init(monkeypatch, registry, boards, tab="boards")

    assert on_tasks[0] == {"selected": "tasks"}
    assert on_boards[0] == {"selected": "leaderboard"}
    assert "<b>5</b> tasks" in on_tasks[3]


def test_a_view_link_sets_its_pills(monkeypatch, registry, boards):
    pills = _init(
        monkeypatch, registry, boards, area="gastroenterology", task="clinical-scores"
    )[4:7]

    assert [pill["value"] for pill in pills] == [
        "bulk-rna",
        "gastroenterology",
        "clinical-scores",
    ]


def test_an_old_board_link_opens_its_view(monkeypatch, registry, boards):
    pills = _init(monkeypatch, registry, boards, board="endotype-bulk-rna")[4:7]

    assert [pill["value"] for pill in pills] == ["bulk-rna", ALL, "endotype"]


def test_an_unknown_pill_falls_back_to_all(monkeypatch, registry, boards):
    pills = _init(monkeypatch, registry, boards, area="nowhere", task="nothing")[4:7]

    assert [pill["value"] for pill in pills] == ["bulk-rna", ALL, ALL]


def test_a_pill_clicked_before_the_page_loads_changes_nothing():
    assert app._render_view(None, "bulk-rna", ALL, ALL, "show", "show") == "skip"


def _counted_builds(monkeypatch, registry: dict) -> list:
    """Record each build of the pages, from ``registry`` and no results."""
    builds = []
    empty = pd.DataFrame(columns=app.RESULT_COLUMNS)

    def page_state(version, revision=None):
        builds.append(version)
        return registry, build_boards(registry), empty, version

    monkeypatch.setattr(app, "_page_state", page_state)
    monkeypatch.setattr(app, "PAGES_BY_VERSION", {})
    return builds


def _unreachable(*args):
    raise ConnectionError("Hub unavailable")


def test_the_pages_are_built_again_only_when_the_results_file_changes(monkeypatch):
    builds = _counted_builds(monkeypatch, dict(REGISTRY))
    hashes = iter([("a", "commit-a"), ("a", "commit-a"), ("b", "commit-b")])
    monkeypatch.setattr(app, "results_hash", lambda token: next(hashes))

    first, second, third = [app._pages(CURRENT) for _ in range(3)]

    assert second is first
    assert third is not first
    assert len(builds) == 2


def test_each_version_keeps_its_own_pages(monkeypatch):
    builds = _counted_builds(monkeypatch, dict(REGISTRY))
    monkeypatch.setattr(app, "results_hash", lambda token: ("a", "commit-a"))

    current, closed = app._pages(CURRENT), app._pages(CLOSED)

    assert current.state[3] == CURRENT
    assert closed.state[3] == CLOSED
    assert app._pages(CURRENT) is current
    assert len(builds) == 2


def test_pages_without_a_registry_are_not_kept(monkeypatch):
    builds = _counted_builds(monkeypatch, {})
    monkeypatch.setattr(app, "results_hash", lambda token: ("a", "commit-a"))

    app._pages(CURRENT)
    app._pages(CURRENT)

    assert len(builds) == 2


def test_the_last_pages_show_when_the_hub_gives_no_hash(monkeypatch):
    builds = _counted_builds(monkeypatch, dict(REGISTRY))
    monkeypatch.setattr(app, "results_hash", lambda token: ("a", "commit-a"))
    kept = app._pages(CURRENT)
    monkeypatch.setattr(app, "results_hash", _unreachable)

    assert app._pages(CURRENT) is kept
    assert len(builds) == 1


def test_the_last_pages_show_when_a_new_build_fails(monkeypatch):
    _counted_builds(monkeypatch, dict(REGISTRY))
    monkeypatch.setattr(app, "results_hash", lambda token: ("a", "commit-a"))
    kept = app._pages(CURRENT)
    builds = _counted_builds(monkeypatch, {})
    app.PAGES_BY_VERSION[CURRENT.number] = kept
    monkeypatch.setattr(app, "results_hash", lambda token: ("b", "commit-b"))

    assert app._pages(CURRENT) is kept
    assert app._pages(CURRENT) is kept
    assert len(builds) == 2


def test_pages_built_without_a_hash_are_not_kept(monkeypatch):
    builds = _counted_builds(monkeypatch, dict(REGISTRY))
    monkeypatch.setattr(app, "results_hash", _unreachable)

    app._pages(CURRENT)
    app._pages(CURRENT)

    assert len(builds) == 2
    assert app.PAGES_BY_VERSION == {}


def _fake_hub(monkeypatch) -> None:
    """Serve the test registry and no results from the Hub, at hash ``a``."""
    monkeypatch.delenv("PRIMO_FIXTURE", raising=False)
    tasks = list(REGISTRY.values())
    datasets = [{"id": task["dataset_id"]} for task in tasks]
    monkeypatch.setattr(app, "fetch_manifest", lambda token, revision: datasets)
    monkeypatch.setattr(app, "fetch_tasks_registry", lambda token, revision: tasks)
    monkeypatch.setattr(app, "fetch_public_task_ids", lambda token, revision: set())
    monkeypatch.setattr(
        app,
        "read_results",
        lambda token, revision: pd.DataFrame(columns=app.RESULT_COLUMNS),
    )
    monkeypatch.setattr(app, "results_hash", lambda token: ("a", "commit-a"))
    monkeypatch.setattr(app, "PAGES_BY_VERSION", {})


def test_the_results_are_read_at_the_commit_of_their_hash(monkeypatch):
    _fake_hub(monkeypatch)
    revisions = []
    empty = pd.DataFrame(columns=app.RESULT_COLUMNS)
    monkeypatch.setattr(
        app, "read_results", lambda token, revision: revisions.append(revision) or empty
    )

    app._pages(CURRENT)

    assert revisions == ["commit-a"]


@pytest.mark.parametrize("fetch", ["read_results", "fetch_public_task_ids"])
def test_pages_built_from_a_failed_fetch_are_not_kept(monkeypatch, fetch):
    _fake_hub(monkeypatch)
    kept = app._pages(CURRENT)
    monkeypatch.setattr(app, fetch, _unreachable)
    monkeypatch.setattr(app, "results_hash", lambda token: ("b", "commit-b"))

    assert app._pages(CURRENT) is kept
    assert app.PAGES_BY_VERSION == {CURRENT.number: kept}


def test_a_view_is_rendered_once_per_results_file(monkeypatch):
    _counted_builds(monkeypatch, dict(REGISTRY))
    monkeypatch.setattr(app, "results_hash", lambda token: ("a", "commit-a"))
    rendered = []
    monkeypatch.setattr(
        app, "render_view", lambda *args: rendered.append(args) or "view"
    )
    pages = app._pages(CURRENT)

    for hvg in ("show", "show", "hide"):
        assert app._render_view(pages, "bulk-rna", ALL, ALL, hvg, "show") == "view"

    assert len(rendered) == 2


def test_a_page_load_gives_its_pages_to_the_pills(monkeypatch, registry, boards):
    state = _init(monkeypatch, registry, boards)[9]

    assert isinstance(state, app.Pages)
    assert state.state[0] is registry


def test_the_public_tasks_pill_shows_only_when_a_task_is_public(
    monkeypatch, registry, boards
):
    private = {task_id: {**task, "public": False} for task_id, task in registry.items()}

    shown = _init(monkeypatch, registry, boards)[8]
    hidden = _init(monkeypatch, private, boards)[8]

    assert shown["visible"] is True
    assert hidden["visible"] is False


def test_the_page_loads_no_stylesheet_from_another_site():
    css = Path(app.CSS_PATH).read_text()

    assert 'rel="stylesheet"' not in app.PAGE_HEAD
    assert "@import" not in css


def test_submission_also_saves_its_hvg_concatenation_entry(monkeypatch):
    persisted = []

    summary = _evaluate(monkeypatch, lambda rows, token: persisted.extend(rows))

    by_name = {row["model_name"]: row for row in persisted}
    assert set(by_name) == {"eva-rna-v1", "eva-rna-v1 + HVG"}
    plain, concatenated = by_name["eva-rna-v1"], by_name["eva-rna-v1 + HVG"]
    for column in ("hf_username", "institution", "submission_id", "submitted_at"):
        assert plain[column] == concatenated[column]
    assert "So is **eva-rna-v1 + HVG**" in summary


def test_hvg_failure_still_saves_the_plain_entry(monkeypatch):
    persisted = []

    summary = _evaluate(
        monkeypatch, lambda rows, token: persisted.extend(rows), hvg_fails=True
    )

    assert {row["model_name"] for row in persisted} == {"eva-rna-v1"}
    assert "eva-rna-v1 is now shown on the relevant boards" in summary
    assert "`eva-rna-v1 + HVG` entry could not be scored" in summary


def test_the_hvg_entry_does_not_repeat_the_missing_datasets(monkeypatch):
    partial = {**_result(), "full_coverage": False, "missing": ["d902"]}

    summary = _evaluate(monkeypatch, lambda rows, token: None, result=partial)

    assert "**eva-rna-v1 + HVG**: covered" in summary
    assert summary.count("missing from file: ['d902']") == 1


def test_hvg_tag_is_reserved(monkeypatch):
    persisted = []

    summary = _evaluate(
        monkeypatch, lambda rows, token: persisted.extend(rows), model="eva + hvg"
    )

    assert persisted == []
    assert "is reserved" in summary


def _check_the_space_build(check: str) -> None:
    """Build the interface in a fresh Python, then run ``check`` on ``demo``.

    Outside a Space, the Gradio login needs a local token, so the env fakes a Space.
    """
    space_env = {
        "SYSTEM": "spaces",
        "SPACE_ID": "owner/space",
        "OAUTH_CLIENT_ID": "client-id",
        "OAUTH_CLIENT_SECRET": "client-secret",
        "OAUTH_SCOPES": "openid profile",
        "OPENID_PROVIDER_URL": "https://huggingface.co",
    }
    subprocess.run(
        [sys.executable, "-c", f"import app; demo = app.build_demo(); {check}"],
        cwd=Path(app.__file__).parent,
        env={**os.environ, **space_env},
        check=True,
    )


def test_the_interface_builds_as_on_the_space():
    """The web form and the ``/submit`` endpoint share one queue."""
    _check_the_space_build(
        "queues = {fn.api_name: fn.concurrency_id for fn in demo.fns.values()}; "
        "assert queues['submit'] == queues['evaluate'] == app.SUBMISSIONS, queues"
    )


def test_the_top_bar_links_reload_after_an_evaluation():
    assert "window.pmReload" in app.TABLE_SORT_JS
    _check_the_space_build(
        "fns = list(demo.fns.values()); "
        "evaluate = next(fn for fn in fns if fn.api_name == 'evaluate'); "
        "after = [fn.js for fn in fns if fn.trigger_after == evaluate._id]; "
        "assert app.RELOAD_AFTER_EVALUATION_JS in after, after"
    )


def test_the_page_shows_one_loader_until_the_first_page_load_ends():
    css = Path(app.CSS_PATH).read_text()

    assert "html:not(.pm-ready) #pm-site { visibility: hidden; }" in css
    assert "html.pm-ready #pm-loader { display: none !important; }" in css
    _check_the_space_build(
        "blocks = demo.blocks.values(); "
        "assert any(getattr(b, 'elem_id', None) == 'pm-loader' for b in blocks); "
        "fns = list(demo.fns.values()); "
        "init = next(fn for fn in fns if fn.name == '_init'); "
        "assert init.show_progress == 'hidden', init.show_progress; "
        "after = [fn.js for fn in fns if fn.trigger_after == init._id]; "
        "assert app.PAGE_READY_JS in after, after"
    )


def test_a_page_shows_only_the_rows_of_the_current_version(monkeypatch):
    monkeypatch.setattr(app, "_registry_by_id", lambda version: dict(REGISTRY))
    ledger = _ledger(("legacy", "alice", "t901"), ("current", "bob", "t901"))
    ledger["version"] = ["", CURRENT.number]
    monkeypatch.setattr(app, "read_results", lambda token, revision: ledger)

    _, _, rows, version = app._page_state()

    assert list(rows["model_name"]) == ["current"]
    assert version == CURRENT


def test_a_visitor_can_ask_for_a_closed_version(monkeypatch):
    monkeypatch.setattr(app, "VERSIONS", [CURRENT, CLOSED])

    assert app._viewed_version({"version": CLOSED.number}) == CLOSED
    assert app._viewed_version({"version": "9.9"}) == CURRENT
    assert app._viewed_version({}) == CURRENT


def test_submit_always_shows_the_current_version(monkeypatch):
    monkeypatch.setattr(app, "VERSIONS", [CURRENT, CLOSED])

    params = {"version": CLOSED.number, "tab": "submit"}

    assert app._viewed_version(params) == CURRENT


def test_a_closed_version_reads_its_commits_and_its_rows(monkeypatch):
    monkeypatch.delenv("PRIMO_FIXTURE", raising=False)
    revisions = []

    def fetch(value):
        return lambda token, revision: revisions.append(revision) or value

    monkeypatch.setattr(app, "fetch_manifest", fetch([{"id": "d901"}]))
    tasks = [{"task_id": "t901", "dataset_id": "d901"}]
    monkeypatch.setattr(app, "fetch_tasks_registry", fetch(tasks))
    monkeypatch.setattr(app, "fetch_public_task_ids", fetch(set()))
    ledger = _ledger(("old", "alice", "t901"), ("new", "bob", "t901"))
    ledger["version"] = [CLOSED.number, CURRENT.number]
    monkeypatch.setattr(app, "read_results", lambda token, revision: ledger)

    by_id, _, rows, version = app._page_state(CLOSED)

    assert revisions == [CLOSED.data, CLOSED.labels, CLOSED.data]
    assert list(by_id) == ["t901"]
    assert list(rows["model_name"]) == ["old"]
    assert version == CLOSED


def test_the_theme_loads_the_regular_and_bold_fonts_of_the_design():
    assert app.THEME["font"][0] == ("IBM Plex Sans", (400, 700))
    assert app.THEME["font_mono"][0] == ("IBM Plex Mono", (400, 700))


def test_the_page_style_goes_to_launch_as_gradio_6_asks():
    keywords = {
        node.func.attr: {keyword.arg for keyword in node.keywords}
        for node in ast.walk(ast.parse(Path(app.__file__).read_text()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") in ("Blocks", "launch")
    }
    style = {"theme", "head", "css_paths", "js"}

    assert style <= keywords["launch"]
    assert not style & keywords["Blocks"]
