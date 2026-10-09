"""Boards: the task sets models are ranked on.

Each modality is a board, and so is each therapeutic area and each task family
within a modality, so models of two modalities are never ranked together. The
"Other" area mixes unrelated diseases: it gets no board, but its tasks still
count on their modality and family boards.
"""

from dataclasses import dataclass

from primomics.evaluate import TASK_ID
from primomics.submission import norm_id

MODALITY = "modality"
THERAPEUTIC_AREA = "therapeutic_area"
CATEGORY = "category"
OTHER_AREA = "Other"


@dataclass(frozen=True)
class Board:
    """One leaderboard: the tasks whose ``group`` field is ``name``.

    ``group`` is ``modality``, ``therapeutic_area`` or ``category``; the last two
    are scoped to ``modality``.
    """

    group: str
    name: str
    modality: str
    task_ids: frozenset[str]


def _distinct(tasks: list[dict], key: str) -> list[str]:
    """Distinct non-empty values of ``key``, sorted case-insensitively."""
    return sorted({str(task[key]) for task in tasks if task.get(key)}, key=str.lower)


def build_boards(tasks: list[dict]) -> list[Board]:
    """Every board the tasks support: each modality, then its areas and families."""
    boards = []
    for modality in _distinct(tasks, MODALITY):
        within = [task for task in tasks if str(task.get(MODALITY)) == modality]
        boards.append(
            Board(
                MODALITY,
                modality,
                modality,
                frozenset(norm_id(task[TASK_ID]) for task in within),
            )
        )
        for group in (THERAPEUTIC_AREA, CATEGORY):
            for name in _distinct(within, group):
                if group == THERAPEUTIC_AREA and name == OTHER_AREA:
                    continue
                boards.append(
                    Board(
                        group,
                        name,
                        modality,
                        frozenset(
                            norm_id(task[TASK_ID])
                            for task in within
                            if str(task.get(group)) == name
                        ),
                    )
                )
    return boards
