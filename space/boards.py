"""The slices of PRIMO a visitor can rank models on: registry tasks -> boards.

A *board* is a self-contained leaderboard over a subset of the registry -- every
task of a modality, of a therapeutic area, or of a task family. Coverage is
judged INSIDE a board, so a model that covered every Oncology task is ranked
on the Oncology board even though it skipped Infectious diseases. Each board is
a real leaderboard, not a view filter.

Modality stays the wall for every board, so models from different omics layers
are never ranked together. The leaderboard page picks a board with three pills:
modality, therapeutic area and task family. ``view_board`` builds the board of
any such view, area and family crossed included.

``OPEN_BOARDS`` names the slices PRIMO does NOT cover, so the page states its own
gaps instead of implying the registry is the whole territory. They are a
hand-written constant, not a roadmap: a slice belongs here when a contributor
could plausibly bring it, and it disappears on its own the day the registry
covers it.

Modalities and therapeutic areas get open slices; task categories deliberately do
not. A category is an enum whose members are each pinned to a single metric, so
naming an open one ("biomarker discovery") would advertise a scoring rule that
does not exist. Modalities and areas only need a cohort.

Pure functions over registry dicts -- no Gradio, no network, no HTML.
"""

import re
from dataclasses import dataclass, replace

from evaluator import _norm_id

METRIC_LABEL = {
    "auroc": "AUROC",
    "pearson": "Pearson",
    "residual_spearman": "Residual Spearman score",
    "centered_spearman": "Centered Spearman score",
}
MODALITY_LABEL = {
    "bulk RNA": "Bulk RNAseq",
    "Single Cell RNA": "Single-cell RNAseq",
    "single-cell RNA": "Single-cell RNAseq",
}

MODALITY_GROUP = "Per modality"
AREA_GROUP = "Per therapeutic areas"
CATEGORY_GROUP = "Task Category"
OTHER_AREA = "Other"
PUBLIC = "public"

CODE_LENGTH = 3
FALLBACK_CODE = "n/a"
DISPLAY_LABEL = {"perturbation_prediction": "Perturbation response"}

CATEGORY_BLURB = {
    "treatment_outcome": "Will this patient respond to the drug?",
    "clinical_scores": "How severe is this patient's disease?",
    "prognosis": "How will this patient's disease turn out?",
    "endotype": "Which molecular subtype is this patient?",
    "perturbation_prediction": "How will treatment change this patient's expression?",
}

MAX_LISTED = 3


def label(value: str) -> str:
    """``treatment_outcome`` -> ``Treatment outcome``; leaves free text alone."""
    return DISPLAY_LABEL.get(value, value.replace("_", " ").capitalize())


def metric_label(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric)


def modality_label(modality: str) -> str:
    """Return the visitor-facing label for a stored modality identifier."""
    return MODALITY_LABEL.get(modality, modality)


def short_code(name: str) -> str:
    """The board's letter tag, standing in for what used to be a per-board emoji.

    Derived from the name rather than looked up, because a lookup table is what
    breaks: spatial transcriptomics or metabolomics would land on a shrug the day
    somebody adds them. Colour carries the group; these letters only carry the
    board.
    """
    letters = re.sub(r"[^a-z]", "", name.lower())
    return letters[:CODE_LENGTH].upper() or FALLBACK_CODE


def slugify(*parts: str) -> str:
    """URL-safe key for a board, stable enough to paste into a link."""
    joined = "-".join(str(p) for p in parts)
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", joined.lower())).strip("-")


def distinct(tasks, key: str) -> list[str]:
    """Distinct non-empty values of ``key`` across registry tasks, case-insensitive."""
    return sorted({str(t[key]) for t in tasks if t.get(key)}, key=str.lower)


def distinct_listed(tasks, key: str) -> list[str]:
    """Distinct values of a list-valued ``key`` (a cohort can span several diseases)."""
    values: set[str] = set()
    for task in tasks:
        values.update(str(value) for value in task.get(key) or [])
    return sorted(values, key=str.lower)


@dataclass(frozen=True)
class Board:
    """One enterable leaderboard: a named task set plus the numbers on its card."""

    slug: str
    group: str
    name: str
    code: str
    blurb: str
    modality: str
    task_ids: frozenset[str]
    n_cohorts: int
    n_patients: int
    n_diseases: int

    @property
    def n_tasks(self) -> int:
        return len(self.task_ids)


def _join(values: list[str]) -> str:
    """``a, b and c``, truncated -- card blurbs must not wrap forever.

    A truncated list drops the "and": ``a, b and c…`` reads as an ellipsis stuck
    to ``c``, where ``a, b, c…`` reads as the list continuing.
    """
    shown = values[:MAX_LISTED]
    if not shown:
        return "n/a"
    if len(values) > MAX_LISTED:
        return f"{', '.join(shown)}…"
    if len(shown) == 1:
        return shown[0]
    return f"{', '.join(shown[:-1])} and {shown[-1]}"


def _cohort_stats(tasks: list[dict]) -> tuple[int, int, int]:
    """Cohorts, patients and diseases behind a task set.

    Counted per COHORT, not per task and not per dataset. Three clinical scores
    read off the same biopsies are one cohort, and so are the two treatment arms
    and the transfer split cut from one anti-TNF trial -- summing datasets would
    advertise those patients twice. Entries on one cohort can differ in
    ``n_samples`` (labels get dropped, an arm is a subset), so the widest one
    stands for it.

    A registry written before ``cohort_id`` existed falls back to ``dataset_id``,
    which is the old behaviour rather than a crash.
    """
    patients_by_cohort: dict[str, int] = {}
    diseases: set[str] = set()
    for task in tasks:
        cohort = str(task.get("cohort_id") or _norm_id(task.get("dataset_id", "")))
        n_subjects = int(task.get("n_subjects") or task.get("n_samples") or 0)
        patients_by_cohort[cohort] = max(patients_by_cohort.get(cohort, 0), n_subjects)
        diseases.update(str(d) for d in task.get("diseases") or [])
    return (
        len(patients_by_cohort),
        sum(patients_by_cohort.values()),
        len(diseases),
    )


def _blurb(group: str, name: str, tasks: list[dict]) -> str:
    if group == MODALITY_GROUP:
        display = modality_label(name)
        return f"Every PRIMO task, scored from {display[:1].lower()}{display[1:]} profiles."
    if group == CATEGORY_GROUP:
        return CATEGORY_BLURB.get(name, label(name))
    families = [label(c) for c in distinct(tasks, "category")]
    return f"{_join(families)} across {_join(distinct_listed(tasks, 'diseases'))}."


def _board(group: str, name: str, modality: str, tasks: list[dict]) -> Board:
    """Build one board from a task slice."""
    n_cohorts, n_patients, n_diseases = _cohort_stats(tasks)
    display = modality_label(name) if group == MODALITY_GROUP else label(name)
    return Board(
        slug=slugify(modality) if group == MODALITY_GROUP else slugify(name, modality),
        group=group,
        name=display,
        code=short_code(display),
        blurb=_blurb(group, name, tasks),
        modality=modality,
        task_ids=frozenset(_norm_id(t["task_id"]) for t in tasks),
        n_cohorts=n_cohorts,
        n_patients=n_patients,
        n_diseases=n_diseases,
    )


def without_public(board: Board, by_id: dict[str, dict]) -> Board:
    """The same board restricted to its private tasks, card numbers recounted.

    Parameters
    ----------
    board : Board
        Board built from ``by_id``.
    by_id : dict[str, dict]
        Task registry keyed by task id, public tasks flagged.

    Returns
    -------
    Board
        The board with the same slug and name; its task set may be empty.
    """
    private = frozenset(t for t in board.task_ids if not by_id[t].get(PUBLIC))
    n_cohorts, n_patients, n_diseases = _cohort_stats([by_id[t] for t in private])
    return replace(
        board,
        task_ids=private,
        n_cohorts=n_cohorts,
        n_patients=n_patients,
        n_diseases=n_diseases,
    )


def build_boards(by_id: dict[str, dict]) -> list[Board]:
    """Every board the registry supports, modality boards first.

    A facet value that yields no task simply yields no card, so a registry
    written before therapeutic areas existed degrades to modality boards only
    instead of breaking the page.

    The "Other" area gets no board: it mixes unrelated diseases, so its ranking
    would mean nothing. Its tasks are still ranked on the modality and category
    boards.
    """
    boards: list[Board] = []
    for modality in distinct(by_id.values(), "modality"):
        within = [t for t in by_id.values() if str(t.get("modality")) == modality]
        boards.append(_board(MODALITY_GROUP, modality, modality, within))
        for area in distinct(within, "therapeutic_area"):
            if area == OTHER_AREA:
                continue
            tasks = [t for t in within if str(t.get("therapeutic_area")) == area]
            boards.append(_board(AREA_GROUP, area, modality, tasks))
        for category in distinct(within, "category"):
            tasks = [t for t in within if str(t.get("category")) == category]
            boards.append(_board(CATEGORY_GROUP, category, modality, tasks))
    return boards


@dataclass(frozen=True)
class OpenBoard:
    """A slice nobody can be ranked on yet: a stated gap, not a leaderboard.

    Deliberately not a ``Board``: it has no tasks, no cohorts and no patients, and
    zeroing those fields would print "0 patients" where the page invites a cohort.
    """

    group: str
    name: str
    blurb: str


OPEN_BOARDS: tuple[OpenBoard, ...] = (
    OpenBoard(
        MODALITY_GROUP,
        "Single-cell RNAseq",
        "Dissociated tissue, labelled at the patient level. No cohort yet.",
    ),
    OpenBoard(
        MODALITY_GROUP,
        "Proteomics",
        "Plasma or tissue proteins paired with clinical follow-up. No cohort yet.",
    ),
    OpenBoard(
        MODALITY_GROUP,
        "Spatial transcriptomics",
        "Expression kept in place in the tissue, with patient outcomes. No cohort yet.",
    ),
    OpenBoard(
        AREA_GROUP,
        "Neuroscience",
        "Neurodegenerative cohorts, such as Alzheimer's or Parkinson's. No cohort yet.",
    ),
    OpenBoard(
        AREA_GROUP,
        "Cardiovascular",
        "Heart or vessel disease with outcomes or treatment response. No cohort yet.",
    ),
)


def in_group(boards: list[Board], group: str) -> list[Board]:
    """The boards of one group, biggest first."""
    return sorted(
        (b for b in boards if b.group == group), key=lambda b: (-b.n_tasks, b.name)
    )


def open_in_group(boards: list[Board], group: str) -> list[OpenBoard]:
    """Open slices of one group, minus any slice the registry has since covered.

    The day a single-cell cohort lands, its board is built from the registry and
    the matching open slice drops out with no edit here.
    """
    covered = {board.name.lower() for board in in_group(boards, group)}
    return [
        board
        for board in OPEN_BOARDS
        if board.group == group and board.name.lower() not in covered
    ]


ALL = "all"


def _within(by_id: dict[str, dict], modality: str) -> list[dict]:
    """The tasks of the modality whose slug is ``modality``."""
    return [t for t in by_id.values() if slugify(str(t.get("modality"))) == modality]


def modality_choices(by_id: dict[str, dict]) -> list[tuple[str, str]]:
    """One ``(label, slug)`` pill per modality of the registry."""
    choices = {
        slugify(m): modality_label(m) for m in distinct(by_id.values(), "modality")
    }
    return [(name, slug) for slug, name in choices.items()]


def area_choices(by_id: dict[str, dict], modality: str) -> list[tuple[str, str]]:
    """The therapeutic-area pills of a modality, "All" first and no "Other"."""
    areas = distinct(_within(by_id, modality), "therapeutic_area")
    return [("All", ALL)] + [(a, slugify(a)) for a in areas if a != OTHER_AREA]


def category_choices(by_id: dict[str, dict], modality: str) -> list[tuple[str, str]]:
    """The task-category pills of a modality, "All" first."""
    categories = distinct(_within(by_id, modality), "category")
    return [("All", ALL)] + [(label(c), slugify(c)) for c in categories]


def view_board(
    by_id: dict[str, dict], modality: str, area: str, category: str
) -> Board:
    """The board of one view: a modality's tasks, narrowed by area and category."""
    within = _within(by_id, modality)
    tasks = [
        t
        for t in within
        if area in (ALL, slugify(str(t.get("therapeutic_area"))))
        and category in (ALL, slugify(str(t.get("category"))))
    ]
    raw_modality = str(within[0]["modality"]) if within else modality
    if area == ALL and category == ALL:
        return _board(MODALITY_GROUP, raw_modality, raw_modality, tasks)
    if category == ALL:
        name = str(tasks[0]["therapeutic_area"]) if tasks else area
        return _board(AREA_GROUP, name, raw_modality, tasks)
    name = str(tasks[0]["category"]) if tasks else category
    return _board(CATEGORY_GROUP, name, raw_modality, tasks)


def board_view(board: Board, by_id: dict[str, dict]) -> tuple[str, str, str]:
    """The ``(modality, area, category)`` pills that show an old ``?board=`` link."""
    task = next((by_id[t] for t in sorted(board.task_ids) if t in by_id), {})
    area = ALL
    category = ALL
    if board.group == AREA_GROUP and task:
        area = slugify(str(task.get("therapeutic_area")))
    if board.group == CATEGORY_GROUP and task:
        category = slugify(str(task.get("category")))
    return slugify(board.modality), area, category
