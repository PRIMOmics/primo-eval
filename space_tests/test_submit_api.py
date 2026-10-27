"""Tests of the ``/submit`` endpoint that ``primo submit`` calls (no network)."""

import ast
import json
import logging
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import app
import gradio as gr
import huggingface_hub
import pandas as pd
import pytest
import requests
from gradio_client import Client, handle_file
from huggingface_hub.utils import HfHubHTTPError
from submit_api import (
    ERROR,
    FILE_DATA,
    MAX_FILE_SIZE,
    PUBLISHED,
    REFUSED,
    SUBMISSIONS_PER_DAY,
    SUBMITTED_AT,
    route,
)

CLIENT = Path(__file__).parents[1] / "src" / "primomics" / "submit.py"
ALICE = "hf_alice_secret_901"
BOB = "hf_bob_secret_902"
SPACE_TOKEN = "hf_space_secret_909"
ACCOUNTS = {ALICE: "alice", BOB: "bob"}
UPLOAD = {"path": "upload/submission.parquet", "meta": {"_type": FILE_DATA}}
FIELDS = {
    "submission_path": UPLOAD,
    "modality": "bulk-rna",
    "model_name": "model-901",
    "institution": "",
    "is_author_submission": False,
    "email": "person@example.org",
    "paper_link": "",
    "hf_model_link": "",
    "notes": "",
    "token": ALICE,
}


def _client() -> ast.Module:
    return ast.parse(CLIENT.read_text())


def _client_fields() -> list[str]:
    """The names that ``primo submit`` sends, read from its own code."""
    predict = next(
        node
        for node in ast.walk(_client())
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "predict"
    )
    return [keyword.arg for keyword in predict.keywords if keyword.arg != "api_name"]


def _client_constant(name: str):
    """A constant of ``primo submit``, read from its own code."""
    return next(
        ast.literal_eval(node.value)
        for node in _client().body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )


def _row(model, owner, submission_id, submitted_at, is_baseline=False) -> dict:
    return {
        "model_name": model,
        "hf_username": owner,
        "task_id": "t901",
        "score": 0.5,
        "submitted_at": submitted_at,
        "is_baseline": is_baseline,
        "submission_id": submission_id,
    }


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


@pytest.fixture
def space(monkeypatch):
    """The Space on a fake Hub, which records what a submission would save."""
    saved = SimpleNamespace(
        ledger=pd.DataFrame([_row("other-901", "carol", "s-0", "2026-01-01")]),
        whoami=[],
        scored=[],
        archived=[],
        rows=[],
        meta=[],
    )

    class Api:
        def __init__(self, token=None):
            pass

        def whoami(self, token=None):
            saved.whoami.append(token)
            if token not in ACCOUNTS:
                raise HfHubHTTPError(
                    f"401 Unauthorized: invalid token {token}",
                    response=SimpleNamespace(status_code=401, headers={}, request=None),
                )
            return {"name": ACCOUNTS[token]}

    def score_all(path, token, modality, representation, reference=None):
        saved.scored.append(path)
        return _result()

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(app, "_results_frame", lambda: saved.ledger)
    monkeypatch.setattr(app, "score_all", score_all)
    monkeypatch.setattr(app, "fetch_hvg_reference", lambda token: "hvg.parquet")
    monkeypatch.setattr(
        app, "archive_submission", lambda *args: saved.archived.append(args)
    )
    monkeypatch.setattr(
        app, "append_results", lambda rows, token: saved.rows.extend(rows)
    )
    monkeypatch.setattr(
        app, "append_submission", lambda meta, token: saved.meta.append(meta)
    )
    return saved


def test_the_endpoint_takes_exactly_the_fields_of_primo_submit(space):
    assert sorted(_client_fields()) == sorted(FIELDS)

    reply = route(app.run_submission)(
        **{name: FIELDS[name] for name in _client_fields()}
    )

    assert sorted(reply) == sorted(_client_constant("REPLY_KEYS"))
    assert reply["status"] == _client_constant("PUBLISHED") == PUBLISHED
    assert _client_constant("REFUSED") == REFUSED
    assert reply["submission_id"]
    assert set(space.scored) == {UPLOAD["path"]}
    assert {row["hf_username"] for row in space.rows} == {"alice"}


def test_a_name_held_by_another_account_is_refused(space):
    space.ledger = pd.DataFrame([_row("model-901", "bob", "s-1", "2026-01-01")])

    reply = route(app.run_submission)(**FIELDS)

    assert reply["status"] == REFUSED
    assert "already submitted by another HF account" in reply["message"]
    assert "bob" not in reply["message"]
    assert space.scored == space.archived == space.rows == space.meta == []


def test_the_token_is_never_kept_or_shown(space, capsys, caplog):
    unknown = "hf_unknown_secret_903"
    endpoint = route(app.run_submission)

    with caplog.at_level(logging.DEBUG):
        published = endpoint(**FIELDS)
        refused = endpoint(**{**FIELDS, "token": unknown})

    assert published["status"] == PUBLISHED
    assert refused["status"] == REFUSED
    printed = capsys.readouterr()
    kept = [published, refused, space.scored, space.archived, space.rows, space.meta]
    seen = repr(kept) + printed.out + printed.err + caplog.text
    assert ALICE not in seen
    assert unknown not in seen


@pytest.fixture
def hub(monkeypatch):
    """The real ``whoami`` on a fake Hub, in a Space that holds ``SPACE_TOKEN``.

    The Hub records the token of each request and gives the ``answer`` it is set
    to: an HTTP status, or an error to raise instead.
    """
    monkeypatch.setenv("HF_TOKEN", SPACE_TOKEN)
    known = {**ACCOUNTS, SPACE_TOKEN: "space-owner"}
    hub = SimpleNamespace(tokens=[], answer=HTTPStatus.OK)

    def send(session, request, **kwargs):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        hub.tokens.append(token)
        if isinstance(hub.answer, Exception):
            raise hub.answer
        response = requests.Response()
        response.status_code = hub.answer
        response._content = json.dumps({"name": known.get(token)}).encode()
        response.request, response.url = request, request.url
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return hub


def _core(*fields) -> dict:
    return {"status": PUBLISHED, "message": f"as {fields[-1]}", "submission_id": "s"}


@pytest.mark.parametrize("token", ["", "  ", None, True, 1, ["hf_x"], {"name": "x"}])
def test_only_a_token_string_reaches_whoami(hub, token):
    """For any other value, ``whoami`` falls back to the token of the Space."""
    answer = route(_core)(**{**FIELDS, "token": token})

    assert answer["status"] == REFUSED
    assert hub.tokens == []


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        (HTTPStatus.OK, PUBLISHED),
        (HTTPStatus.UNAUTHORIZED, REFUSED),
        (HTTPStatus.FORBIDDEN, REFUSED),
        (HTTPStatus.TOO_MANY_REQUESTS, ERROR),
        (HTTPStatus.SERVICE_UNAVAILABLE, ERROR),
        (requests.ConnectionError("no network"), ERROR),
        (requests.ReadTimeout("too slow"), ERROR),
    ],
)
def test_only_a_refused_token_is_the_callers_fault(hub, capsys, caplog, answer, status):
    hub.answer = answer

    with caplog.at_level(logging.DEBUG):
        result = route(_core)(**FIELDS)

    assert result["status"] == status
    assert hub.tokens == [ALICE]
    printed = capsys.readouterr()
    assert ALICE not in repr(result) + printed.out + printed.err + caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        "/etc/hosts",
        {"path": "/etc/hosts"},
        {"path": "/etc/hosts", "meta": {"_type": "other"}},
    ],
)
def test_a_file_without_the_upload_marker_is_refused(space, payload):
    """Gradio checks only the paths that carry the marker of an upload."""
    reply = route(app.run_submission)(**{**FIELDS, "submission_path": payload})

    assert reply["status"] == REFUSED
    assert space.whoami == space.scored == []


@pytest.mark.parametrize(
    ("published_today", "status"),
    [(SUBMISSIONS_PER_DAY - 1, PUBLISHED), (SUBMISSIONS_PER_DAY, REFUSED)],
)
def test_an_account_publishes_a_few_submissions_per_day(space, published_today, status):
    now = datetime.now(UTC)
    today = (now - timedelta(hours=1)).strftime(SUBMITTED_AT)
    before = (now - timedelta(days=2)).strftime(SUBMITTED_AT)
    rows = [_row("model-901", "alice", "s-old", before)]
    for index in range(published_today):
        rows.append(_row("model-901", "alice", f"s-{index}", today))
    rows.append(_row("model-901 + HVG", "alice", "s-0", today))
    rows.append(_row("HVG-1200-genes", "alice", "", today, is_baseline=True))
    space.ledger = pd.DataFrame(rows)
    endpoint = route(app.run_submission)

    alice = endpoint(**FIELDS)
    bob = endpoint(**{**FIELDS, "model_name": "model-902", "token": BOB})

    assert alice["status"] == status
    assert bob["status"] == PUBLISHED
    if status == REFUSED:
        assert f"{SUBMISSIONS_PER_DAY} submissions in 24 hours" in alice["message"]


def test_primo_submit_knows_the_file_limit_of_the_space():
    assert _client_constant("MAX_FILE_BYTES") == MAX_FILE_SIZE


def test_the_space_launches_with_the_file_limit():
    launch = next(
        node
        for node in ast.walk(ast.parse(Path(app.__file__).read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "launch"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in launch.keywords}
    assert keywords.get("max_file_size") == "MAX_FILE_SIZE"


@pytest.fixture
def served(space):
    """The endpoint on a local Gradio server, which checks the files it receives."""
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.api(route(app.run_submission), api_name="submit")
    _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True)
    yield url
    demo.close()


def test_a_file_sent_like_primo_submit_reaches_the_scorer(served, space, tmp_path):
    sent = tmp_path / "submission.parquet"
    sent.write_bytes(b"embeddings of model-901")
    client = Client(
        served, verbose=False, download_files=False, analytics_enabled=False
    )

    reply = client.predict(
        **{**FIELDS, "submission_path": handle_file(str(sent))}, api_name="/submit"
    )

    assert reply["status"] == PUBLISHED
    [scored] = set(space.scored)
    assert Path(scored) != sent
    assert Path(scored).name == sent.name
    assert Path(scored).read_bytes() == sent.read_bytes()


def test_gradio_refuses_a_marked_path_that_was_not_uploaded(served, space, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("a file of the Space")
    marked = {"path": str(secret), "meta": {"_type": FILE_DATA}}
    call = f"{served}gradio_api/call/submit"

    data = list({**FIELDS, "submission_path": marked}.values())
    event = requests.post(call, json={"data": data}, timeout=30).json()["event_id"]
    answer = requests.get(f"{call}/{event}", timeout=30).text

    assert "event: error" in answer
    assert space.whoami == space.scored == []
