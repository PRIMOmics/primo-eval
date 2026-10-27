"""Unit tests for board construction (no network, no Gradio)."""

from boards import (
    ALL,
    AREA_GROUP,
    CATEGORY_GROUP,
    MODALITY_GROUP,
    OPEN_BOARDS,
    area_choices,
    board_view,
    build_boards,
    category_choices,
    in_group,
    metric_label,
    modality_choices,
    open_in_group,
    short_code,
    view_board,
    without_public,
)


def test_modality_board_holds_every_task_of_that_modality(named, registry):
    assert named("bulk RNA").task_ids == set(registry)


def test_area_and_category_boards_each_partition_the_modality(boards, registry):
    for group in (AREA_GROUP, CATEGORY_GROUP):
        covered = [tid for b in in_group(boards, group) for tid in b.task_ids]
        assert sorted(covered) == sorted(registry)


def test_patients_are_counted_per_cohort_not_per_task(named):
    """t902 and t903 read off the same biopsies -- one cohort, not two."""
    gastro = named("Gastroenterology")
    assert gastro.n_tasks == 3
    assert gastro.n_cohorts == 2
    assert gastro.n_patients == 90 + 120


def test_board_counts_roll_up_to_the_modality(named):
    board = named("bulk RNA")
    assert (board.n_cohorts, board.n_patients, board.n_diseases) == (4, 350, 4)


def test_board_counts_unique_subjects_when_samples_are_repeated(registry):
    registry["t905"]["n_samples"] = 126
    registry["t905"]["n_subjects"] = 20
    board = next(b for b in build_boards(registry) if b.name == "Rheumatology")
    assert board.n_patients == 20


def test_datasets_cut_from_the_same_patients_are_counted_once(registry):
    """Two arms of one trial are two dataset_ids but one group of people."""
    registry["t905"]["cohort_id"] = registry["t901"]["cohort_id"]
    board = next(b for b in build_boards(registry) if b.name == "Bulk RNAseq")
    assert board.n_cohorts == 3
    assert board.n_patients == 90 + 120 + 80


def test_a_registry_without_cohorts_still_counts_per_dataset(registry):
    """Falling back to dataset_id is the old behaviour, not a crash."""
    for task in registry.values():
        del task["cohort_id"]
    board = next(b for b in build_boards(registry) if b.name == "Bulk RNAseq")
    assert (board.n_cohorts, board.n_patients) == (4, 350)


def test_every_board_never_spans_two_modalities(registry):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "single-cell",
    }
    for board in build_boards(registry):
        modalities = {registry[t]["modality"] for t in board.task_ids}
        assert modalities == {board.modality}


def test_a_second_modality_gets_distinct_area_and_category_boards(registry, boards):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "single-cell",
    }
    grown = build_boards(registry)
    assert len(in_group(grown, MODALITY_GROUP)) == 2
    assert len(grown) == len(boards) + 3
    rheumatology = [b for b in grown if b.name == "Rheumatology"]
    assert {b.modality for b in rheumatology} == {"bulk RNA", "single-cell"}
    endotype = [b for b in grown if b.name == "Endotype"]
    assert {b.modality for b in endotype} == {"bulk RNA", "single-cell"}
    assert {b.task_ids for b in endotype} == {frozenset({"t905"}), frozenset({"t906"})}
    assert {b.slug for b in endotype} == {"endotype-bulk-rna", "endotype-single-cell"}


def test_slugs_are_unique_and_url_safe(boards):
    slugs = [b.slug for b in boards]
    assert len(set(slugs)) == len(slugs)
    assert all(s.replace("-", "").isalnum() for s in slugs)


def test_registry_without_areas_degrades_to_modality_boards(registry):
    for task in registry.values():
        del task["therapeutic_area"]
    boards = build_boards(registry)
    assert in_group(boards, AREA_GROUP) == []
    assert [b.name for b in in_group(boards, MODALITY_GROUP)] == ["Bulk RNAseq"]


def test_empty_registry_yields_no_boards():
    assert build_boards({}) == []


def test_short_codes_come_from_the_name_not_a_lookup_table():
    """A modality nobody planned for still gets a tag instead of a fallback."""
    assert short_code("bulk RNA") == "BUL"
    assert short_code("Single Cell RNA") == "SIN"
    assert short_code("Treatment outcome") == "TRE"
    assert short_code("spatial metabolomics") == "SPA"
    assert short_code("") == "n/a"
    assert short_code("42 (!)") == "n/a"


def test_every_board_carries_a_code(boards):
    assert all(board.code.isalpha() and len(board.code) == 3 for board in boards)


def _declared(group: str) -> list[str]:
    return [b.name for b in OPEN_BOARDS if b.group == group]


def test_open_boards_are_the_slices_the_registry_lacks(boards):
    for group in (MODALITY_GROUP, AREA_GROUP):
        names = [b.name for b in open_in_group(boards, group)]
        assert names == _declared(group)
    assert "bulk RNA" not in _declared(MODALITY_GROUP)
    assert "Gastroenterology" not in _declared(AREA_GROUP)


def test_an_open_board_drops_out_when_the_registry_covers_it(registry):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "Single Cell RNA",
    }
    boards = build_boards(registry)
    live_names = [b.name for b in in_group(boards, MODALITY_GROUP)]
    names = [b.name for b in open_in_group(boards, MODALITY_GROUP)]

    assert live_names == ["Bulk RNAseq", "Single-cell RNAseq"]
    assert "Single-cell RNAseq" not in names
    assert len(names) == len(_declared(MODALITY_GROUP)) - 1


def test_live_registry_single_cell_spelling_removes_open_card(registry):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "modality": "Single Cell RNA",
        "therapeutic_area": "Infectious diseases",
    }
    boards = build_boards(registry)
    assert any(board.name == "Single-cell RNAseq" for board in boards)
    assert any(board.name == "Infectious diseases" for board in boards)
    assert "Single-cell RNAseq" not in {
        board.name for board in open_in_group(boards, MODALITY_GROUP)
    }


def test_an_open_area_drops_out_when_a_cohort_lands_there(registry):
    registry["t906"] = {
        **registry["t905"],
        "task_id": "t906",
        "therapeutic_area": "Neuroscience",
    }
    names = [b.name for b in open_in_group(build_boards(registry), AREA_GROUP)]
    assert "Neuroscience" not in names
    assert names == ["Cardiovascular"]


def test_other_area_gets_no_board_but_its_tasks_stay_ranked(registry):
    """A board mixing unrelated diseases would rank nothing."""
    registry["t905"]["therapeutic_area"] = "Other"
    boards = build_boards(registry)
    assert "Other" not in {b.name for b in in_group(boards, AREA_GROUP)}
    assert "t905" in next(b for b in boards if b.name == "Bulk RNAseq").task_ids
    assert "t905" in next(b for b in boards if b.name == "Endotype").task_ids


def test_task_categories_never_get_an_open_card(boards):
    """A category is pinned to one metric, so an open one would promise a probe."""
    assert open_in_group(boards, CATEGORY_GROUP) == []
    assert _declared(CATEGORY_GROUP) == []


def test_an_empty_registry_still_states_every_open_slice():
    assert len(open_in_group([], MODALITY_GROUP)) == len(_declared(MODALITY_GROUP))
    assert len(open_in_group([], AREA_GROUP)) == len(_declared(AREA_GROUP))


def test_a_truncated_list_drops_the_and(registry):
    """``a, b and c…`` reads as an ellipsis stuck to c; ``a, b, c…`` reads as more."""
    from boards import _join

    assert _join(["a", "b"]) == "a and b"
    assert _join(["a", "b", "c"]) == "a, b and c"
    assert _join(["a", "b", "c", "d"]) == "a, b, c…"
    assert _join([]) == "n/a"


def test_blurbs_describe_what_the_board_asks(named):
    assert "bulk RNAseq" in named("bulk RNA").blurb
    assert "respond" in named("Treatment outcome").blurb
    assert named("Clinical scores").blurb == "How severe is this patient's disease?"
    assert "Psoriasis" in named("Dermatology").blurb


def test_perturbation_category_is_a_regular_board_with_centered_spearman(registry):
    registry["t918"] = {
        **registry["t905"],
        "task_id": "t918",
        "category": "perturbation_prediction",
        "metric": "centered_spearman",
    }
    board = next(
        board
        for board in build_boards(registry)
        if board.name == "Perturbation response"
    )
    assert board.task_ids == {"t918"}
    assert "treatment" in board.blurb.lower()
    assert metric_label("centered_spearman") == "Centered Spearman score"


def test_without_public_leaves_the_public_tasks_out_and_recounts(named, registry):
    board = named("bulk RNA")
    private = without_public(board, registry)
    assert private.task_ids == board.task_ids - {"t904"}
    assert (private.slug, private.name) == (board.slug, board.name)
    assert (private.n_cohorts, private.n_patients) == (3, 270)


def test_without_public_empties_a_board_of_public_tasks(named, registry):
    private = without_public(named("Dermatology"), registry)
    assert private.task_ids == frozenset()
    assert (private.n_cohorts, private.n_patients, private.n_diseases) == (0, 0, 0)


# ---------------------------------------------------------------- view pills
def test_a_modality_spelled_twice_gets_one_pill(registry):
    for task_id, spelling in (("t906", "Single Cell RNA"), ("t907", "single-cell RNA")):
        registry[task_id] = {
            **registry["t905"],
            "task_id": task_id,
            "modality": spelling,
        }

    assert modality_choices(registry) == [
        ("Bulk RNAseq", "bulk-rna"),
        ("Single-cell RNAseq", "single-cell-rna"),
    ]


def test_area_and_task_pills_start_with_all_and_skip_other(registry):
    registry["t906"] = {
        **registry["t901"],
        "task_id": "t906",
        "therapeutic_area": "Other",
    }

    assert area_choices(registry, "bulk-rna") == [
        ("All", ALL),
        ("Dermatology", "dermatology"),
        ("Gastroenterology", "gastroenterology"),
        ("Rheumatology", "rheumatology"),
    ]
    assert category_choices(registry, "bulk-rna") == [
        ("All", ALL),
        ("Clinical scores", "clinical-scores"),
        ("Endotype", "endotype"),
        ("Treatment outcome", "treatment-outcome"),
    ]


def test_a_view_crosses_an_area_with_a_task_family(registry):
    crossed = view_board(registry, "bulk-rna", "gastroenterology", "clinical-scores")

    assert crossed.task_ids == {"t902", "t903"}
    assert view_board(registry, "bulk-rna", ALL, ALL).task_ids == set(registry)
    assert not view_board(registry, "bulk-rna", "dermatology", "endotype").task_ids


def test_a_view_of_one_axis_holds_the_tasks_of_its_board(boards, registry):
    for board in boards:
        view = view_board(registry, *board_view(board, registry))
        assert view.task_ids == board.task_ids


def test_an_old_board_link_maps_to_its_pills(boards, registry):
    by_slug = {board.slug: board for board in boards}

    assert board_view(by_slug["bulk-rna"], registry) == ("bulk-rna", ALL, ALL)
    assert board_view(by_slug["gastroenterology-bulk-rna"], registry) == (
        "bulk-rna",
        "gastroenterology",
        ALL,
    )
    assert board_view(by_slug["endotype-bulk-rna"], registry) == (
        "bulk-rna",
        ALL,
        "endotype",
    )
