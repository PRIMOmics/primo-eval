"""Failures, split by who has to fix them."""


class SubmissionError(ValueError):
    """A submission the evaluator cannot score (bad format, missing samples...)."""


class EvaluatorError(RuntimeError):
    """A failure on our side (data fetch, unexpected bug), not the submitter's.

    Raised instead of being recorded as a per-dataset skip, so a transient
    Hugging Face hiccup or an evaluator bug never silently costs a submitter
    their coverage.
    """
