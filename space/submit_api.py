"""The ``/submit`` endpoint that ``primo submit`` calls.

The web form and the endpoint share one path: both call ``run_submission`` in
``app.py``, so the same checks, scores and records apply to both. The endpoint
reads the caller's Hugging Face token only to learn the account name, with
``whoami``. It never stores the token, logs it or puts it in a message. Only a
non-empty string reaches ``whoami``: for any other value, such as ``""``, ``None``
or ``True``, ``whoami`` falls back to the token of the Space itself. A 401 or a
403 from ``whoami`` means a bad token, which the caller fixes. Any other failure,
such as a timeout or a 5xx, is on our side.

Gradio checks that a file comes from an upload only when its payload carries the
``gradio.FileData`` marker, which ``gradio_client.handle_file`` adds. Without the
marker, a caller could send the path of any file of the Space, so the endpoint
refuses such a payload.

An account can publish ``SUBMISSIONS_PER_DAY`` submissions in 24 hours, through
the form and the endpoint together. A file can weigh ``MAX_FILE_SIZE`` at most.
The form and the endpoint share the ``SUBMISSIONS`` queue, so one submission runs
at a time.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from http import HTTPStatus

import pandas as pd
from results import (
    IS_BASELINE,
    OWNER,
    SUBMISSION_ID,
    with_baseline_flag,
    with_owner,
    with_submission_id,
)

PUBLISHED = "published"
REFUSED = "refused"
ERROR = "error"
SUBMISSIONS = "submissions"
SUBMISSIONS_PER_DAY = 4
MAX_FILE_SIZE = 500_000_000
SUBMITTED_AT = "%Y-%m-%d %H:%M:%S.%f"
FILE_DATA = "gradio.FileData"


def reply(status: str, message: str, submission_id: str = "") -> dict:
    """The answer to one submission, as ``primo submit`` reads it."""
    return {"status": status, "message": message, "submission_id": submission_id}


def recent_submissions(df: pd.DataFrame, owner: str, now: datetime) -> int:
    """How many submissions ``owner`` published in the 24 hours before ``now``."""
    since = (now - timedelta(days=1)).strftime(SUBMITTED_AT)
    rows = with_submission_id(with_owner(with_baseline_flag(df)))
    recent = rows[
        (rows[OWNER] == owner)
        & ~rows[IS_BASELINE]
        & (rows["submitted_at"].astype(str) >= since)
    ]
    return int(recent[SUBMISSION_ID].nunique())


def route(core: Callable[..., dict]) -> Callable[..., dict]:
    """The ``/submit`` endpoint, which checks, scores and saves through ``core``."""

    def submit(
        submission_path: dict,
        modality: str,
        model_name: str,
        institution: str,
        is_author_submission: bool,
        email: str,
        paper_link: str,
        hf_model_link: str,
        notes: str,
        token: str,
    ) -> dict:
        """Score one submission file that ``primo submit`` sends."""
        from huggingface_hub import HfApi

        meta = (
            submission_path.get("meta") if isinstance(submission_path, dict) else None
        )
        if not isinstance(meta, dict) or meta.get("_type") != FILE_DATA:
            return reply(
                REFUSED, "Send the submission file with gradio_client.handle_file."
            )
        if not isinstance(token, str) or not token.strip():
            return reply(REFUSED, "Send a Hugging Face token: run `hf auth login`.")
        try:
            username = HfApi().whoami(token=token.strip())["name"]
        except Exception as error:  # noqa: BLE001
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                return reply(
                    REFUSED,
                    "Hugging Face refused the token. Log in again with "
                    "`hf auth login`.",
                )
            logging.warning(
                "whoami failed: %s, status %s", type(error).__name__, status
            )
            return reply(
                ERROR,
                "⚠️ We could not check the token with Hugging Face. This is on our "
                "side, not your token. Please try again in a moment.",
            )
        return core(
            submission_path["path"],
            modality,
            model_name,
            institution,
            is_author_submission,
            email,
            paper_link,
            hf_model_link,
            notes,
            username,
        )

    return submit
