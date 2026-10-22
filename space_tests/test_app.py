"""Unit tests for submission feedback in the PRIMO app."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
from boards import build_boards
from render import render_boards
from versions import CURRENT

from .conftest import CLOSED, REGISTRY

previous_gradio = sys.modules.get("gradio")
gradio_stub = ModuleType("gradio")
gradio_stub.themes = SimpleNamespace(Base=lambda **kwargs: object())
gradio_stub.OAuthProfile = object
gradio_stub.Request = object
gradio_stub.Blocks = object
gradio_stub.Tabs = lambda selected: selected
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
    monkeypatch.setattr(app, "_rendered_board", lambda slug: "board")

    summary, _ = app.evaluate(
        "submission.csv",
        "bulk-rna",
        model,
        "Scienta",
        True,
        "person@example.org",
        "",
        "",
        "",
        None,
        SimpleNamespace(username="submitter"),
    )
    return summary


def test_successful_submission_confirms_model_is_on_relevant_boards(monkeypatch):
    summary = _evaluate(monkeypatch, lambda rows, token: None)

    assert "eva-rna-v1 is now shown on the relevant boards" in summary


def test_table_sort_prefers_the_numeric_cell_value_over_sd_display_text():
    assert "cell.dataset.sortValue || cell.textContent.trim()" in app.TABLE_SORT_JS


def test_owned_model_name_does_not_reveal_owner(monkeypatch):
    monkeypatch.setattr(app, "_results_frame", lambda: _ledger())
    monkeypatch.setattr(app, "_claimed_by", lambda df, model: "private-owner")

    summary, _ = app.evaluate(
        "submission.csv",
        "bulk-rna",
        "eva-rna-v1",
        "Scienta",
        True,
        "person@example.org",
        "",
        "",
        "",
        None,
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


def test_overview_renders_from_the_state_init_hands_over(monkeypatch):
    by_id = {task_id: dict(task) for task_id, task in REGISTRY.items()}
    boards = build_boards(by_id)
    df = _ledger(("eva", "alice", "t901"), ("eva", "alice", "t902"))
    monkeypatch.setattr(
        app, "_page_state", lambda version: (by_id, boards, df, version)
    )
    monkeypatch.setattr(app.gr, "Tabs", lambda **kwargs: kwargs, raising=False)

    state = app._init(None)[3]

    assert app._render_overview(state) == (render_boards(boards, df, by_id), None)


def test_only_the_board_page_renders_a_board(monkeypatch, registry, boards):
    empty = pd.DataFrame(columns=app.RESULT_COLUMNS)
    monkeypatch.setattr(
        app, "_page_state", lambda version: (registry, boards, empty, version)
    )
    monkeypatch.setattr(app.gr, "Tabs", lambda **kwargs: kwargs, raising=False)

    on_tasks = app._init(SimpleNamespace(query_params={"tab": "tasks"}))
    on_board = app._init(SimpleNamespace(query_params={"board": "bulk-rna"}))

    assert on_tasks[-1] == ""
    assert "<h1>Bulk RNAseq</h1>" in on_board[-1]


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


def test_hvg_tag_is_reserved(monkeypatch):
    persisted = []

    summary = _evaluate(
        monkeypatch, lambda rows, token: persisted.extend(rows), model="eva + hvg"
    )

    assert persisted == []
    assert "is reserved" in summary


def test_the_interface_builds_as_on_the_space():
    """Outside a Space, the Gradio login needs a local token, so the env fakes a Space."""
    space_env = {
        "SYSTEM": "spaces",
        "SPACE_ID": "owner/space",
        "OAUTH_CLIENT_ID": "client-id",
        "OAUTH_CLIENT_SECRET": "client-secret",
        "OAUTH_SCOPES": "openid profile",
        "OPENID_PROVIDER_URL": "https://huggingface.co",
    }
    subprocess.run(
        [sys.executable, "-c", "import app; app.build_demo()"],
        cwd=Path(app.__file__).parent,
        env={**os.environ, **space_env},
        check=True,
    )


def test_a_page_shows_only_the_rows_of_the_current_version(monkeypatch):
    monkeypatch.setattr(app, "_registry_by_id", lambda version: dict(REGISTRY))
    ledger = _ledger(("legacy", "alice", "t901"), ("current", "bob", "t901"))
    ledger["version"] = ["", CURRENT.number]
    monkeypatch.setattr(app, "read_results", lambda token: ledger)

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
    monkeypatch.setattr(app, "read_results", lambda token: ledger)

    by_id, _, rows, version = app._page_state(CLOSED)

    assert revisions == [CLOSED.data, CLOSED.labels, CLOSED.data]
    assert list(by_id) == ["t901"]
    assert list(rows["model_name"]) == ["old"]
    assert version == CLOSED
