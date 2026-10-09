from primomics.boards import (
    CATEGORY,
    MODALITY,
    OTHER_AREA,
    THERAPEUTIC_AREA,
    Board,
    build_boards,
)

TASKS = [
    {
        "task_id": "t901",
        "modality": "bulk RNA",
        "therapeutic_area": "Oncology",
        "category": "treatment_outcome",
    },
    {
        "task_id": "t902",
        "modality": "bulk RNA",
        "therapeutic_area": "Oncology",
        "category": "prognosis",
    },
    {
        "task_id": "t903",
        "modality": "bulk RNA",
        "therapeutic_area": "dermatology",
        "category": "treatment_outcome",
    },
    {
        "task_id": "t904",
        "modality": "bulk RNA",
        "therapeutic_area": OTHER_AREA,
        "category": "endotype",
    },
    {
        "task_id": "t905",
        "modality": "bulk RNA",
        "therapeutic_area": None,
        "category": "clinical_scores",
    },
    {
        "task_id": "t906",
        "modality": "sc RNA",
        "therapeutic_area": "Oncology",
        "category": "treatment_outcome",
    },
]


def _board(group, name, modality, *task_ids):
    return Board(group, name, modality, frozenset(task_ids))


def test_each_modality_comes_first_then_its_areas_then_its_families():
    assert build_boards(TASKS) == [
        _board(
            MODALITY, "bulk RNA", "bulk RNA", "t901", "t902", "t903", "t904", "t905"
        ),
        _board(THERAPEUTIC_AREA, "dermatology", "bulk RNA", "t903"),
        _board(THERAPEUTIC_AREA, "Oncology", "bulk RNA", "t901", "t902"),
        _board(CATEGORY, "clinical_scores", "bulk RNA", "t905"),
        _board(CATEGORY, "endotype", "bulk RNA", "t904"),
        _board(CATEGORY, "prognosis", "bulk RNA", "t902"),
        _board(CATEGORY, "treatment_outcome", "bulk RNA", "t901", "t903"),
        _board(MODALITY, "sc RNA", "sc RNA", "t906"),
        _board(THERAPEUTIC_AREA, "Oncology", "sc RNA", "t906"),
        _board(CATEGORY, "treatment_outcome", "sc RNA", "t906"),
    ]


def test_no_board_spans_two_modalities():
    modality_of = {task["task_id"]: task["modality"] for task in TASKS}

    for board in build_boards(TASKS):
        assert {modality_of[task_id] for task_id in board.task_ids} == {board.modality}


def test_the_other_area_gets_no_board_but_its_tasks_stay_ranked():
    boards = build_boards(TASKS)

    assert OTHER_AREA not in {board.name for board in boards}
    assert [board.name for board in boards if "t904" in board.task_ids] == [
        "bulk RNA",
        "endotype",
    ]


def test_a_task_without_an_area_stays_on_its_modality_and_family_boards():
    boards = build_boards(TASKS)

    assert [board.name for board in boards if "t905" in board.task_ids] == [
        "bulk RNA",
        "clinical_scores",
    ]


def test_board_task_ids_are_normalized():
    (board,) = build_boards([{"task_id": " t901 ", "modality": "bulk RNA"}])

    assert board.task_ids == {"t901"}


def test_no_tasks_no_boards():
    assert build_boards([]) == []
