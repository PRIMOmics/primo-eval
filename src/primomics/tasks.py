"""Task registries: the public task specs shipped with the package, and a reader."""

from collections import Counter
from pathlib import Path

import yaml

from primomics.errors import EvaluatorError
from primomics.evaluate import TASK_ID
from primomics.submission import norm_id

PUBLIC_TASKS = Path(__file__).with_name("public_tasks.yaml")


def load_tasks_registry(path: str | Path) -> list[dict]:
    """Read a task registry, one entry per task; no task or a duplicate id is a bug."""
    with open(path) as handle:
        data = yaml.safe_load(handle)
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not tasks:
        raise EvaluatorError(f"no tasks in the registry: {path}")
    ids = [norm_id(t[TASK_ID]) for t in tasks]
    duplicates = sorted(task_id for task_id, seen in Counter(ids).items() if seen > 1)
    if duplicates:
        raise EvaluatorError(f"duplicate task_id in the registry: {duplicates}")
    return tasks
