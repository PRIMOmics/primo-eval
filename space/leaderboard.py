"""Table building for the PRIMO front-end: persisted results + registry -> frames.

Kept apart from Gradio and from any I/O so it unit-tests without a network
(``app.py`` fetches and renders, ``boards.py`` decides what a board *is*).

``ranked_table`` lists only models that covered EVERY task the Elo anchor
scored, so a partial run cannot win by skipping the hard cohorts.
``per_task_table`` keeps partial submissions and is never a ranking.

Our baselines rank where their score puts them: pinning one to the bottom would
hide a foundation model losing to an information-rich expression baseline. Task
metadata is read defensively, so an old registry yields blank cells rather than
a broken page.
"""

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from boards import (
    AREA_GROUP,
    MODALITY_GROUP,
    PUBLIC,
    Board,
    label,
    metric_label,
    modality_label,
)
from evaluator import _norm_id
from results import (
    INSTITUTION,
    IS_AUTHOR_SUBMISSION,
    IS_BASELINE,
    MODEL_NAME,
    PAPER_LINK,
    SUBMISSION_ID,
    display_name,
    split_hvg_name,
    with_author_submission_flag,
    with_baseline_flag,
    with_institution,
    with_paper_link,
    with_submission_id,
)
from scipy.optimize import minimize
from scipy.special import expit
from scoring import TaskScore, category_means, sort_key

EMPTY_RANKED_COLUMNS = ["Model"]
EMPTY_PER_TASK_COLUMNS = ["Task"]
SCORE_DECIMALS = 3
ELO_START = 1000.0
ELO_SCALE = 400 / math.log(10)
ELO_REGULARIZATION = 0.1
ELO_ANCHOR_MODEL = "HVG-1200-genes"


@dataclass(frozen=True)
class TopModel:
    """One line of a home card's mini-ranking; ``elo`` is None when unavailable."""

    name: str
    elo: int | None


class ModelCell(str):
    """String-compatible model label with paper-link and chip metadata.

    ``is_hvg`` marks a ``<model> + HVG`` entry: the label is the parent name and
    the tag is rendered as a chip.
    """

    paper_link: str
    is_author_submission: bool
    is_hvg: bool

    def __new__(
        cls,
        label: str,
        paper_link: str,
        is_author_submission: bool,
        is_hvg: bool = False,
    ) -> "ModelCell":
        cell = super().__new__(cls, label)
        cell.paper_link = paper_link
        cell.is_author_submission = is_author_submission
        cell.is_hvg = is_hvg
        return cell


class TaskCell(str):
    """String-compatible task title flagged when its labels are public."""

    is_public: bool

    def __new__(cls, title: str, is_public: bool) -> "TaskCell":
        cell = super().__new__(cls, title)
        cell.is_public = is_public
        return cell


def _task_cell(task: dict) -> TaskCell:
    """The task's title, or its id when it has none."""
    return TaskCell(
        str(task.get("title") or task.get("task_id", "")), bool(task.get(PUBLIC))
    )


def _round(value: float | None) -> float | None:
    """Round for display; anything non-finite becomes a blank cell, not ``-inf``."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, SCORE_DECIMALS)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the credit columns and add the tiebreak key rows are ordered by.

    ``_submission_key`` falls back to the timestamp for rows written before
    ``submission_id`` existed, so legacy rows still order deterministically.
    """
    flagged = with_submission_id(
        with_paper_link(
            with_author_submission_flag(with_institution(with_baseline_flag(df)))
        )
    )
    if flagged.empty:
        return flagged
    key = flagged[SUBMISSION_ID].mask(
        flagged[SUBMISSION_ID] == "", "legacy:" + flagged["submitted_at"].astype(str)
    )
    return flagged.assign(
        _submission_key=key, _task=flagged["task_id"].map(_norm_id)
    ).sort_values(["submitted_at", "_submission_key"])


def latest_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps the most recent result for each ``(name, is_baseline, task)`` slot.

    An entry is the accumulated set of its slots, not one submission: a later
    submission fills the tasks it covers and leaves the rest standing.

    Parameters
    ----------
    df : pd.DataFrame
        Persisted result rows.

    Returns
    -------
    pd.DataFrame
        One row per slot, credit columns normalised.
    """
    ordered = _ordered(df)
    if ordered.empty:
        return ordered
    return (
        ordered.groupby([MODEL_NAME, IS_BASELINE, "_task"], as_index=False)
        .tail(1)
        .drop(columns=["_submission_key", "_task"])
    )


def submission_counts(df: pd.DataFrame) -> dict[tuple[str, bool], int]:
    """Counts how many times each entry's most-rewritten task was submitted.

    Parameters
    ----------
    df : pd.DataFrame
        Persisted result rows.

    Returns
    -------
    dict[tuple[str, bool], int]
        Count keyed by ``(model_name, is_baseline)``.
    """
    flagged = with_baseline_flag(df)
    if flagged.empty:
        return {}
    per_task = (
        flagged.assign(_task=flagged["task_id"].map(_norm_id))
        .groupby([MODEL_NAME, IS_BASELINE, "_task"])
        .size()
    )
    return {
        (str(model), bool(is_baseline)): int(count)
        for (model, is_baseline), count in per_task.groupby(
            level=[MODEL_NAME, IS_BASELINE]
        )
        .max()
        .items()
    }


def scores_from_rows(rows: pd.DataFrame, by_id: dict[str, dict]) -> list[TaskScore]:
    """Rebuilds one submission's task scores from persisted rows and the registry.

    Parameters
    ----------
    rows : pd.DataFrame
        Result rows of a single entry.
    by_id : dict[str, dict]
        Task registry keyed by task id.

    Returns
    -------
    list[TaskScore]
        One score per row, joined by ``task_id``. Retired tasks are dropped.
    """
    out = []
    for _, r in rows.iterrows():
        task_id = _norm_id(r["task_id"])
        meta = by_id.get(task_id)
        if meta is None:
            continue
        out.append(
            TaskScore(
                task_id=task_id,
                dataset_id=_norm_id(meta["dataset_id"]),
                category=str(meta["category"]),
                metric=str(meta["metric"]),
                score=float(r["score"]),
                n_samples=0,
            )
        )
    return out


def _board_registry(by_id: dict[str, dict], board: Board) -> dict[str, dict]:
    return {tid: meta for tid, meta in by_id.items() if tid in board.task_ids}


def _stored_repeat_scores(row: pd.Series) -> tuple[float, ...]:
    """Read finite persisted repeat scores, or return an empty tuple."""
    raw = row.get("repeat_scores")
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list) or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in parsed
            ):
                return ()
            values = tuple(float(value) for value in parsed)
            if values and all(math.isfinite(value) for value in values):
                return values
        except (
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
            json.JSONDecodeError,
        ):
            pass
    return ()


def _repeat_scores(row: pd.Series) -> tuple[float, ...]:
    """Read persisted repeat scores, falling back to one legacy task score."""
    values = _stored_repeat_scores(row)
    if values:
        return values
    return (float(row["score"]),)


def _paired_splits(
    left: tuple[float, ...], right: tuple[float, ...]
) -> list[tuple[float, ...]]:
    """Lines two entries up split by split; the shorter record plays its mean.

    Pairwise, never board-wide: aligning on the longest record on the board
    would flatten two three-split entries the day a six-split one is published.
    """
    splits = max(len(left), len(right))
    return [
        column if len(column) == splits else (float(np.mean(column)),) * splits
        for column in (left, right)
    ]


def _is_anchor(model: str, is_baseline: bool) -> bool:
    """Is this the Elo anchor? Matched on ``(name, is_baseline)``, not the name."""
    return is_baseline and model == ELO_ANCHOR_MODEL


def _credit_row(rows: pd.DataFrame) -> pd.Series:
    """The most recent slot carrying an institution or a paper link."""
    credited = rows[(rows[INSTITUTION] != "") | (rows[PAPER_LINK] != "")]
    return (credited if not credited.empty else rows).iloc[-1]


def _entries(df: pd.DataFrame, by_id: dict[str, dict], board: Board) -> list[dict]:
    """Rank-ready entries for every model covering the board's scored tasks.

    The anchor decides which tasks count. Without it every finitely-scored task
    counts and the Elo scale falls back to mean-centred, decided per board.
    """
    scoped = _board_registry(by_id, board)
    latest = latest_only(df)
    if latest.empty:
        return []
    on_board = latest[latest["task_id"].map(_norm_id).isin(list(board.task_ids))]
    anchor_rows = on_board[
        [
            _is_anchor(str(model), bool(flag))
            for model, flag in zip(on_board[MODEL_NAME], on_board[IS_BASELINE])
        ]
    ]
    ranked_rows = on_board if anchor_rows.empty else anchor_rows
    scored_tasks = {
        _norm_id(row["task_id"])
        for _, row in ranked_rows.iterrows()
        if math.isfinite(float(row["score"]))
    }
    if not scored_tasks:
        return []

    counts = submission_counts(df)
    entries = []
    for (model, is_baseline), rows in latest.groupby([MODEL_NAME, IS_BASELINE]):
        submitted = rows["submitted_at"].max()
        credits = _credit_row(rows)
        task_rows = rows[
            rows["task_id"].map(_norm_id).isin(list(scored_tasks))
            & rows["score"].map(lambda score: math.isfinite(float(score)))
        ]
        if len(task_rows["task_id"].map(_norm_id).unique()) != len(scored_tasks):
            continue
        categories = category_means(scores_from_rows(task_rows, scoped))
        task_scores = {
            _norm_id(row["task_id"]): float(row["score"])
            for _, row in task_rows.iterrows()
        }
        repeat_scores = {
            _norm_id(row["task_id"]): _repeat_scores(row)
            for _, row in task_rows.iterrows()
        }
        entries.append(
            {
                "model_name": model,
                "institution": credits[INSTITUTION],
                "paper_link": credits[PAPER_LINK],
                "is_author_submission": bool(credits[IS_AUTHOR_SUBMISSION]),
                "submission_id": credits[SUBMISSION_ID],
                "submitted_at": submitted,
                "n_submissions": counts.get((str(model), bool(is_baseline)), 1),
                "categories": categories,
                "rank": sort_key(categories),
                "is_baseline": bool(is_baseline),
                "task_scores": task_scores,
                "repeat_scores": repeat_scores,
            }
        )
    if not entries:
        return entries
    shared = set.intersection(*(set(entry["task_scores"]) for entry in entries))
    _comparison_stats(entries, shared)
    return sorted(entries, key=lambda entry: (-entry["elo"], -entry["rank"]))


def _comparison_stats(entries: list[dict], task_ids: set[str]) -> None:
    """Add Bradley-Terry Elo, mean rank and mean score to ``entries`` in place.

    Battles are played split by split, a pair's splits sharing weight one per
    task. ``ELO_ANCHOR_MODEL`` is pinned during the fit, not shifted after,
    because the ridge penalty shrinks toward whichever model is pinned. Entries
    are keyed by position: two of them can share a name.
    """
    if not entries:
        return
    indices = list(range(len(entries)))
    ranks: dict[int, list[float]] = {index: [] for index in indices}
    battles: list[tuple[int, int, float, float]] = []
    for task_id in sorted(task_ids):
        scores = [entry["task_scores"][task_id] for entry in entries]
        ordered = sorted(
            indices, key=lambda index: (-scores[index], entries[index]["model_name"])
        )
        for index in indices:
            tied_positions = [
                rank
                for rank, other in enumerate(ordered, start=1)
                if scores[other] == scores[index]
            ]
            ranks[index].append(sum(tied_positions) / len(tied_positions))
        for index in indices:
            for opponent in indices[index + 1 :]:
                left, right = _paired_splits(
                    entries[index]["repeat_scores"][task_id],
                    entries[opponent]["repeat_scores"][task_id],
                )
                for ours, theirs in zip(left, right):
                    outcome = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
                    battles.append((index, opponent, outcome, 1 / len(left)))

    anchor = next(
        (
            index
            for index, entry in enumerate(entries)
            if _is_anchor(entry["model_name"], entry["is_baseline"])
        ),
        None,
    )
    pinned = len(entries) - 1 if anchor is None else anchor
    battle_table = np.array(battles, dtype=float).reshape(-1, 4)
    lefts, rights = battle_table[:, 0].astype(int), battle_table[:, 1].astype(int)
    weights = battle_table[:, 3]
    weighted_outcomes = weights * battle_table[:, 2]

    def objective(theta_free: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.insert(theta_free, pinned, 0.0)
        delta = theta[lefts] - theta[rights]
        loss = np.sum(weights * np.logaddexp(0.0, delta) - weighted_outcomes * delta)
        error = weights * expit(delta) - weighted_outcomes
        gradient = np.bincount(lefts, error, len(entries)) - np.bincount(
            rights, error, len(entries)
        )
        loss += ELO_REGULARIZATION * float(np.dot(theta, theta)) / 2
        gradient += ELO_REGULARIZATION * theta
        return float(loss), np.delete(gradient, pinned)

    if battles and len(entries) > 1:
        fit = minimize(
            objective,
            np.zeros(len(entries) - 1),
            jac=True,
            method="L-BFGS-B",
        )
        theta = np.insert(fit.x, pinned, 0.0) if fit.success else np.zeros(len(entries))
        if anchor is None:
            theta -= theta.mean()
        ratings = ELO_START + ELO_SCALE * theta
    else:
        ratings = np.full(len(entries), ELO_START)
    for index, entry in enumerate(entries):
        entry["elo"] = float(ratings[index])
        entry["mean_rank"] = (
            float(np.mean(ranks[index])) if ranks[index] else float("nan")
        )
        entry["mean_score"] = float(np.mean(list(entry["task_scores"].values())))


def ranked_table(
    df: pd.DataFrame, by_id: dict[str, dict], board: Board
) -> pd.DataFrame:
    """Ranks the models that covered every task of ``board`` the anchor scored.

    Parameters
    ----------
    df : pd.DataFrame
        Persisted result rows.
    by_id : dict[str, dict]
        Task registry keyed by task id.
    board : Board
        Board to rank.

    Returns
    -------
    pd.DataFrame
        One row per ranked model, one column per task category in its native
        metric. Empty when no model has full coverage.
    """
    entries = _entries(df, by_id, board)
    if not entries:
        return pd.DataFrame(columns=EMPTY_RANKED_COLUMNS)

    columns = sorted({cat for e in entries for cat in e["categories"]})
    metric_of: dict[str, str] = {}
    for entry in entries:
        for cat, stats in entry["categories"].items():
            metric_of.setdefault(cat, stats["metric"])

    rows = []
    for position, entry in enumerate(entries, start=1):
        categories = entry["categories"]
        name, is_hvg = split_hvg_name(str(entry["model_name"]))
        model = ModelCell(
            display_name(name, entry["is_baseline"]),
            entry["paper_link"],
            entry["is_author_submission"],
            is_hvg,
        )
        row = {
            "Rank": position,
            "Model": model,
            "Institution": entry["institution"] or None,
            "Elo": round(entry["elo"]),
            "Mean rank": _round(entry["mean_rank"]),
            "Mean score": _round(entry["mean_score"]),
            # Blank for baselines and ``+ HVG`` entries: we republish and re-sync
            # those, so their count measures our release cadence, not anyone tuning.
            "Submissions count": (
                None if entry["is_baseline"] or is_hvg else entry["n_submissions"]
            ),
        }
        for cat in columns:
            header = f"{label(cat)} ({metric_label(metric_of[cat])})"
            row[header] = _round(categories[cat]["mean"]) if cat in categories else None
        rows.append(row)
    table = pd.DataFrame(rows)
    table["Submissions count"] = table["Submissions count"].astype("Int64")
    return table


RESERVED_COLUMNS = frozenset({"Task", "Family", "Area", "Modality", "Metric", "Best"})


def per_task_table(
    df: pd.DataFrame, by_id: dict[str, dict], board: Board
) -> pd.DataFrame:
    """Builds the per-task table: one row per task, one column per model.

    Partial submissions are kept, unlike in ``ranked_table``. Model names become
    column names, hence ``RESERVED_COLUMNS``: a model called ``Task`` would
    overwrite every task title.

    Parameters
    ----------
    df : pd.DataFrame
        Persisted result rows.
    by_id : dict[str, dict]
        Task registry keyed by task id.
    board : Board
        Board whose tasks the table is restricted to.

    Returns
    -------
    pd.DataFrame
        Task rows and model columns ordered by mean over the board, plus a
        ``Best`` column naming the leading model.
    """
    scoped = _board_registry(by_id, board)
    if df.empty or not scoped:
        return pd.DataFrame(columns=EMPTY_PER_TASK_COLUMNS)

    scores: dict[tuple[str, str], float] = {}
    repeat_sd: dict[tuple[str, str], tuple[float, int]] = {}
    for _, r in latest_only(df).iterrows():
        task_id = _norm_id(r["task_id"])
        if task_id in scoped:
            model = display_name(str(r[MODEL_NAME]), bool(r[IS_BASELINE]))
            scores[(task_id, model)] = float(r["score"])
            repeats = _stored_repeat_scores(r)
            if len(repeats) > 1:
                repeat_sd[(task_id, model)] = (float(np.std(repeats)), len(repeats))
    if not scores:
        return pd.DataFrame(columns=EMPTY_PER_TASK_COLUMNS)

    by_model: dict[str, list[float]] = {}
    for (_, model), score in scores.items():
        by_model.setdefault(model, []).append(score)
    models = sorted(by_model, key=lambda m: -sum(by_model[m]) / len(by_model[m]))

    rows = []
    displayed_sd: dict[tuple[int, str], tuple[float, int]] = {}
    for task_id, task in sorted(scoped.items()):
        row = {
            "Task": _task_cell(task),
            "Family": label(str(task.get("category", ""))),
        }
        if board.group != AREA_GROUP:
            row["Area"] = str(task.get("therapeutic_area", ""))
        if board.group != MODALITY_GROUP:
            row["Modality"] = modality_label(str(task.get("modality", "")))
        row["Metric"] = metric_label(str(task.get("metric", "")))
        present = {m: scores[(task_id, m)] for m in models if (task_id, m) in scores}
        for model in models:
            row[model] = _round(present.get(model))
            if (task_id, model) in repeat_sd:
                displayed_sd[(len(rows), model)] = repeat_sd[(task_id, model)]
        row["Best"] = max(present, key=present.get) if present else None
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["repeat_standard_deviations"] = displayed_sd
    return table


def top_models(
    df: pd.DataFrame, by_id: dict[str, dict], board: Board, limit: int
) -> list[TopModel]:
    """Lists the board's best full-coverage entries, baselines ranked in place.

    Parameters
    ----------
    df : pd.DataFrame
        Persisted result rows.
    by_id : dict[str, dict]
        Task registry keyed by task id.
    board : Board
        Board to rank.
    limit : int
        Maximum number of entries to return.

    Returns
    -------
    list[TopModel]
        Leaders in rank order; ``elo`` is None when the rating is unavailable.
    """
    return [
        TopModel(
            name=str(e["model_name"]),
            elo=(
                round(e["elo"])
                if math.isfinite(e["elo"]) and math.isfinite(e["rank"])
                else None
            ),
        )
        for e in _entries(df, by_id, board)[:limit]
    ]


TASK_COLUMNS = [
    "Task",
    "Predicts",
    "Family",
    "Area",
    "Modality",
    "Disease",
    "Tissue",
    "Patients",
    "Metric",
]


def tasks_table(tasks: list[dict]) -> pd.DataFrame:
    """Builds the Tasks page table. Provenance is never a column.

    Parameters
    ----------
    tasks : list[dict]
        Registry tasks.

    Returns
    -------
    pd.DataFrame
        One row per task, sorted by task id, with ``TASK_COLUMNS`` as columns.
    """
    if not tasks:
        return pd.DataFrame(columns=TASK_COLUMNS)
    rows = []
    for task in sorted(tasks, key=lambda t: str(t.get("task_id", ""))):
        rows.append(
            {
                "Task": _task_cell(task),
                "Predicts": str(task.get("description") or ""),
                "Family": label(str(task.get("category", ""))),
                "Area": str(task.get("therapeutic_area", "")),
                "Modality": modality_label(str(task.get("modality", ""))),
                "Disease": ", ".join(str(d) for d in task.get("diseases") or []),
                "Tissue": str(task.get("tissue", "")),
                "Patients": task.get("n_subjects") or task.get("n_samples"),
                "Metric": metric_label(str(task.get("metric", ""))),
            }
        )
    return pd.DataFrame(rows, columns=TASK_COLUMNS)


REPOSITORY_OF_PREFIX = {"GSE": "NCBI GEO", "E-MTAB": "EMBL-EBI ArrayExpress"}


def source_repositories(tasks: list[dict]) -> list[str]:
    """Lists the public archives the cohorts come from, never the accessions.

    An accession like ``GSE000000`` would let anyone download the study and
    recover the labels. The registry's ``license`` field stays out too: one
    unvalidated line per dataset, and some cohorts pool three studies under it.

    Parameters
    ----------
    tasks : list[dict]
        Registry tasks, each optionally carrying a ``sources`` list.

    Returns
    -------
    list[str]
        Archive names, sorted.
    """
    repositories: set[str] = set()
    for task in tasks:
        for source in task.get("sources") or []:
            for prefix, repository in REPOSITORY_OF_PREFIX.items():
                if str(source).upper().startswith(prefix):
                    repositories.add(repository)
    return sorted(repositories)
