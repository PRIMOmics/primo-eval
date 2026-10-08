"""Failures, split by who has to fix them."""


class SubmissionError(ValueError):
    """A submission the evaluator cannot score (bad format, missing samples...)."""
