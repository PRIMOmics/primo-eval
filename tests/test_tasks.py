from collections import defaultdict

import pytest

from primomics.errors import EvaluatorError
from primomics.evaluate import MODALITIES
from primomics.tasks import PUBLIC_TASKS, load_tasks_registry

FIELDS = (
    "task_id",
    "dataset_id",
    "modality",
    "category",
    "therapeutic_area",
    "task_type",
    "metric",
    "n_samples",
)
METRIC_TASK_TYPE = {
    "auroc": "classification",
    "pearson": "regression",
    "residual_spearman": "perturbation",
    "centered_spearman": "perturbation",
}


def test_load_tasks_registry_duplicate_raises(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "tasks:\n"
        "  - {task_id: t901, dataset_id: d901}\n"
        "  - {task_id: t901, dataset_id: d902}\n"
    )
    with pytest.raises(EvaluatorError, match="duplicate"):
        load_tasks_registry(path)


def test_load_tasks_registry_without_tasks_raises(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text("public_tasks:\n  - t901\n")
    with pytest.raises(EvaluatorError, match="no tasks"):
        load_tasks_registry(path)


def test_load_tasks_registry_empty_tasks_raises(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text("tasks:\n")
    with pytest.raises(EvaluatorError, match="no tasks"):
        load_tasks_registry(path)


def test_load_tasks_registry_keeps_every_entry_in_order(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "tasks:\n"
        "- task_id: t902\n"
        "  dataset_id: d901\n"
        "  n_samples: 40\n"
        "- task_id: t901\n"
        "  dataset_id: d901\n"
        "  n_top_de_genes: 600\n"
    )
    assert load_tasks_registry(path) == [
        {"task_id": "t902", "dataset_id": "d901", "n_samples": 40},
        {"task_id": "t901", "dataset_id": "d901", "n_top_de_genes": 600},
    ]


def test_there_are_twenty_public_tasks_with_every_field():
    tasks = load_tasks_registry(PUBLIC_TASKS)
    assert len(tasks) == 20
    for task in tasks:
        assert all(task.get(field) for field in FIELDS), task["task_id"]
        assert task["modality"] in MODALITIES.values(), task["task_id"]


def test_each_public_task_metric_fits_its_task_type():
    for task in load_tasks_registry(PUBLIC_TASKS):
        assert METRIC_TASK_TYPE[task["metric"]] == task["task_type"], task["task_id"]


def test_each_public_family_uses_one_metric():
    metrics = defaultdict(set)
    for task in load_tasks_registry(PUBLIC_TASKS):
        metrics[task["category"]].add(task["metric"])
    assert all(len(found) == 1 for found in metrics.values()), dict(metrics)


def test_public_perturbation_tasks_carry_their_decoder_settings():
    tasks = [
        task
        for task in load_tasks_registry(PUBLIC_TASKS)
        if task["task_type"] == "perturbation"
    ]
    assert tasks
    for task in tasks:
        assert task["pairing"] == "paired", task["task_id"]
        assert task["n_top_de_genes"] > 0, task["task_id"]
