"""Shared fixtures for the Space unit tests -- no network, no Gradio.

One small registry stands in for the real ``tasks.yaml``: two tasks share a
dataset (so per-cohort counting is exercised), three therapeutic areas and
three categories, all on one modality. Every dataset gets its own ``cohort_id``
here; the tests that care about shared cohorts rewrite one. ``t904`` is the one
public task, alone on its dataset and its area.
"""

import pandas as pd
import pytest
from boards import build_boards
from results import IS_BASELINE
from versions import Version

CLOSED = Version(
    number="0.0",
    release="v0.0.0",
    opened="2026-09-01",
    data="d" * 40,
    labels="e" * 40,
    reference="f" * 40,
)

RESULT_ROW = ["model_name", "task_id", "score", "submitted_at"]

REGISTRY = {
    "t901": {
        "task_id": "t901",
        "dataset_id": "d901",
        "cohort_id": "cohort-a",
        "category": "treatment_outcome",
        "metric": "auroc",
        "modality": "bulk RNA",
        "therapeutic_area": "Gastroenterology",
        "diseases": ["Crohn Disease", "Ulcerative Colitis"],
        "tissue": "Digestive System",
        "title": "IBD adalimumab remission",
        "description": "Identify patients who achieve endoscopic remission after adalimumab treatment.",
        "target": "endoscopic remission",
        "n_samples": 90,
        "label": {0: "no", 1: "yes"},
        "sources": ["E-MTAB-0001"],
        "license": "EMBL-EBI Terms of Use",
    },
    "t902": {
        "task_id": "t902",
        "dataset_id": "d902",
        "cohort_id": "cohort-b",
        "category": "clinical_scores",
        "metric": "pearson",
        "modality": "bulk RNA",
        "therapeutic_area": "Gastroenterology",
        "diseases": ["Crohn Disease"],
        "tissue": "Digestive System",
        "title": "Crohn SES-CD",
        "description": "How severe?",
        "n_samples": 120,
        "sources": ["GSE000001"],
        "license": "Public (NCBI GEO)",
    },
    "t903": {
        "task_id": "t903",
        "dataset_id": "d902",
        "cohort_id": "cohort-b",
        "category": "clinical_scores",
        "metric": "pearson",
        "modality": "bulk RNA",
        "therapeutic_area": "Gastroenterology",
        "diseases": ["Crohn Disease"],
        "tissue": "Digestive System",
        "title": "Crohn HBI",
        "description": "How severe?",
        "n_samples": 110,
        "sources": ["GSE000001"],
        "license": "Public (NCBI GEO)",
    },
    "t904": {
        "task_id": "t904",
        "dataset_id": "d905",
        "cohort_id": "cohort-c",
        "category": "clinical_scores",
        "metric": "pearson",
        "modality": "bulk RNA",
        "therapeutic_area": "Dermatology",
        "diseases": ["Psoriasis"],
        "tissue": "Skin Tissue",
        "title": "Psoriasis PASI",
        "description": "How severe?",
        "n_samples": 80,
        "sources": ["GSE000002"],
        "license": "Public (NCBI GEO)",
        "public": True,
    },
    "t905": {
        "task_id": "t905",
        "dataset_id": "d907",
        "cohort_id": "cohort-d",
        "category": "endotype",
        "metric": "auroc",
        "modality": "bulk RNA",
        "therapeutic_area": "Rheumatology",
        "diseases": ["Rheumatoid Arthritis"],
        "tissue": "Blood",
        "title": "RA blood endotype",
        "description": "Which subtype?",
        "n_samples": 60,
        "label": {0: "fibroid", 1: "lymphoid"},
        "sources": ["E-MTAB-0002"],
        "license": "EMBL-EBI Terms of Use",
    },
}


@pytest.fixture
def registry() -> dict[str, dict]:
    return {task_id: dict(task) for task_id, task in REGISTRY.items()}


@pytest.fixture
def boards(registry):
    return build_boards(registry)


@pytest.fixture
def named(boards):
    """Look a board up by its display name, e.g. ``named("Rheumatology")``."""

    def _named(name: str):
        return next(
            board for board in boards if board.name == name or board.modality == name
        )

    return _named


@pytest.fixture
def results():
    """A ``task_results.csv`` frame from ``(model, task_id, score, at)`` rows.

    Deliberately without ``is_baseline``: this is the shape of every row written
    before baselines existed, so the whole suite keeps exercising that path.
    """

    def _results(*rows) -> pd.DataFrame:
        return pd.DataFrame(list(rows), columns=RESULT_ROW)

    return _results


@pytest.fixture
def flagged_results():
    """The same, with the flag: ``(model, task_id, score, at, is_baseline)`` rows."""

    def _flagged(*rows) -> pd.DataFrame:
        return pd.DataFrame(list(rows), columns=[*RESULT_ROW, IS_BASELINE])

    return _flagged
