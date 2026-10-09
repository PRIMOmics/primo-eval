"""Rank models on a board with a Bradley-Terry Elo.

Each model keeps its latest score per task: a new submission replaces only the
tasks it covers. The tasks that count are those the anchor baseline scored, and
a model is ranked only with a finite score on all of them. On each task, two
models meet split by split; a model without split scores plays its task score
on every split. The anchor is held at 1000 Elo.
"""

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from primomics.evaluate import TASK_ID
from primomics.metrics import TaskScore, category_means
from primomics.submission import DATASET_ID, norm_id

MODEL_NAME = "model_name"
IS_BASELINE = "is_baseline"
SUBMISSION_ID = "submission_id"
SUBMITTED_AT = "submitted_at"
SCORE = "score"
REPEAT_SCORES = "repeat_scores"
TRUTHY = ("true", "1")

ELO_START = 1000.0
ELO_SCALE = 400 / math.log(10)
ELO_REGULARIZATION = 0.1
ELO_ANCHOR_MODEL = "HVG-1200-genes"


@dataclass(frozen=True)
class Entry:
    """One ranked model, with the task scores its rating was fitted on."""

    model_name: str
    is_baseline: bool
    elo: float
    mean_rank: float
    categories: dict[str, dict]
    task_scores: dict[str, float]
    repeat_scores: dict[str, tuple[float, ...]]


def with_baseline_flag(results: pd.DataFrame) -> pd.DataFrame:
    """Read ``is_baseline`` as a boolean; rows written without it are not baselines.

    Read back from CSV the column can be bool or the strings ``True``/``False``.
    """
    if IS_BASELINE not in results.columns:
        return results.assign(**{IS_BASELINE: False})
    flags = results[IS_BASELINE].astype(str).str.strip().str.lower().isin(TRUTHY)
    return results.assign(**{IS_BASELINE: flags})


def latest_only(results: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest row of each ``(model_name, is_baseline, task)`` slot.

    An entry is the set of its slots, not one submission: a later submission
    replaces the tasks it covers and leaves the others standing. Rows sent at
    the same time are ordered by submission id.
    """
    flagged = with_baseline_flag(results)
    if flagged.empty:
        return flagged
    if SUBMISSION_ID in flagged.columns:
        ids = flagged[SUBMISSION_ID].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
    else:
        ids = pd.Series("", index=flagged.index)
    key = ids.mask(ids == "", "legacy:" + flagged[SUBMITTED_AT].astype(str))
    return (
        flagged.assign(_submission_key=key, _task=flagged[TASK_ID].map(norm_id))
        .sort_values([SUBMITTED_AT, "_submission_key"])
        .groupby([MODEL_NAME, IS_BASELINE, "_task"], as_index=False)
        .tail(1)
        .drop(columns=["_submission_key", "_task"])
    )


def stored_repeat_scores(row: pd.Series) -> tuple[float, ...]:
    """Return the split scores stored with a result if all finite, else nothing."""
    raw = row.get(REPEAT_SCORES)
    if not isinstance(raw, str) or not raw:
        return ()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in parsed
        ):
            return ()
        values = tuple(float(value) for value in parsed)
    except (ValueError, OverflowError, RecursionError):
        return ()
    if values and all(math.isfinite(value) for value in values):
        return values
    return ()


def _repeat_scores(row: pd.Series) -> tuple[float, ...]:
    """Return the stored split scores, or the task score as a single split."""
    values = stored_repeat_scores(row)
    if values:
        return values
    return (float(row[SCORE]),)


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


def _comparison_stats(entries: list[dict], task_ids: set[str]) -> None:
    """Add Bradley-Terry Elo and mean rank to ``entries`` in place.

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


def rank(results: pd.DataFrame, tasks: list[dict]) -> list[Entry]:
    """Rank the models that scored every task of ``tasks`` the anchor scored.

    ``results`` has one row per model, task and submission: ``model_name``,
    ``task_id``, ``score`` and ``submitted_at``, optionally ``is_baseline``,
    ``submission_id`` and ``repeat_scores`` (a JSON list of split scores).
    ``tasks`` are the board's tasks. Without the anchor, every task with a
    finite score counts and the ratings are centred on 1000 instead. Entries
    come best first, by Elo then by the mean of their family means.
    """
    by_id = {norm_id(task[TASK_ID]): task for task in tasks}
    latest = latest_only(results)
    if latest.empty:
        return []
    on_board = latest[latest[TASK_ID].map(norm_id).isin(list(by_id))]
    anchor_rows = on_board[
        [
            _is_anchor(str(model), bool(flag))
            for model, flag in zip(on_board[MODEL_NAME], on_board[IS_BASELINE])
        ]
    ]
    ranked_rows = on_board if anchor_rows.empty else anchor_rows
    scored_tasks = {
        norm_id(row[TASK_ID])
        for _, row in ranked_rows.iterrows()
        if math.isfinite(float(row[SCORE]))
    }
    if not scored_tasks:
        return []

    entries = []
    for (model, is_baseline), rows in latest.groupby([MODEL_NAME, IS_BASELINE]):
        task_rows = rows[
            rows[TASK_ID].map(norm_id).isin(list(scored_tasks))
            & rows[SCORE].map(lambda score: math.isfinite(float(score)))
        ]
        if len(task_rows[TASK_ID].map(norm_id).unique()) != len(scored_tasks):
            continue
        task_scores = {
            norm_id(row[TASK_ID]): float(row[SCORE]) for _, row in task_rows.iterrows()
        }
        categories = category_means(
            [
                TaskScore(
                    task_id=task_id,
                    dataset_id=norm_id(by_id[task_id][DATASET_ID]),
                    category=str(by_id[task_id]["category"]),
                    metric=str(by_id[task_id]["metric"]),
                    score=score,
                    n_samples=0,
                )
                for task_id, score in task_scores.items()
            ]
        )
        entries.append(
            {
                "model_name": str(model),
                "is_baseline": bool(is_baseline),
                "categories": categories,
                "task_scores": task_scores,
                "repeat_scores": {
                    norm_id(row[TASK_ID]): _repeat_scores(row)
                    for _, row in task_rows.iterrows()
                },
            }
        )
    _comparison_stats(entries, scored_tasks)
    entries.sort(
        key=lambda entry: (
            -entry["elo"],
            -np.mean([stats["mean"] for stats in entry["categories"].values()]),
        )
    )
    return [Entry(**entry) for entry in entries]
