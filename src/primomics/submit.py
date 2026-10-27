"""Send a submission file to the leaderboard Space, which scores every task.

The Space exposes a ``/submit`` endpoint whose inputs carry the names of the web
form's ``/evaluate`` endpoint: ``submission_path``, ``modality``, ``model_name``,
``institution``, ``is_author_submission``, ``email``, ``paper_link``,
``hf_model_link`` and ``notes``, plus ``token``, the caller's Hugging Face token.
They are passed by name, so a renamed or missing input raises an error instead of
filling the wrong field. The Space reads the token only to learn who submits, so a
model name stays with the account that first used it. It replies with a ``status``
(``published``, ``refused`` or ``error``), a ``message`` and a ``submission_id``.
The Space takes a file of ``MAX_FILE_BYTES`` at most.
"""

from pathlib import Path

from gradio_client import Client, handle_file

from primomics.errors import EvaluatorError

SPACE = "PRIMOmics/primo-eval"
SUBMIT_API = "/submit"
PUBLISHED = "published"
REFUSED = "refused"
REPLY_KEYS = ("status", "message", "submission_id")
MAX_FILE_BYTES = 500_000_000


def submit(
    path: str | Path,
    *,
    token: str,
    modality: str,
    model_name: str,
    email: str,
    institution: str = "",
    authors: bool = False,
    paper_link: str = "",
    hf_model_link: str = "",
    notes: str = "",
    space: str = SPACE,
) -> dict:
    """Send one submission file to the Space and return its reply."""
    try:
        client = Client(
            space, verbose=False, download_files=False, analytics_enabled=False
        )
        reply = client.predict(
            submission_path=handle_file(str(path)),
            modality=modality,
            model_name=model_name,
            institution=institution,
            is_author_submission=authors,
            email=email,
            paper_link=paper_link,
            hf_model_link=hf_model_link,
            notes=notes,
            token=token,
            api_name=SUBMIT_API,
        )
    except Exception as error:
        raise EvaluatorError(
            f"could not submit to the Space {space}: {error}"
        ) from error
    if not isinstance(reply, dict) or any(key not in reply for key in REPLY_KEYS):
        raise EvaluatorError(f"unexpected reply from the Space {space}: {reply!r}")
    return reply
