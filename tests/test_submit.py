import pytest

from primomics import submit
from primomics.errors import EvaluatorError


def _send(path):
    path.write_bytes(b"embeddings")
    return submit.submit(
        path,
        token="hf_test",
        modality="bulk-rna",
        model_name="model-901",
        email="me@example.org",
        institution="Lab",
        authors=True,
        paper_link="https://example.org/paper",
        hf_model_link="https://huggingface.co/model-901",
        notes="trained on atlas 901",
    )


def test_submit_sends_the_fields_in_the_endpoint_order(fake_space, tmp_path):
    path = tmp_path / "sub.parquet"

    reply = _send(path)

    assert reply == fake_space.reply
    [call] = fake_space.calls
    assert call["src"] == submit.SPACE
    assert call["api_name"] == submit.SUBMIT_API
    assert call["args"][0]["path"] == str(path)
    assert call["args"][1:] == (
        "bulk-rna",
        "model-901",
        "Lab",
        True,
        "me@example.org",
        "https://example.org/paper",
        "https://huggingface.co/model-901",
        "trained on atlas 901",
        "hf_test",
    )


def test_an_unreachable_space_is_an_evaluator_error(monkeypatch, tmp_path):
    def unreachable(src, **kwargs):
        raise ConnectionError("Space asleep")

    monkeypatch.setattr(submit, "Client", unreachable)

    with pytest.raises(EvaluatorError, match="Space asleep"):
        _send(tmp_path / "sub.parquet")


def test_a_malformed_reply_is_an_evaluator_error(fake_space, tmp_path):
    fake_space.reply = "done"

    with pytest.raises(EvaluatorError, match="unexpected reply"):
        _send(tmp_path / "sub.parquet")
