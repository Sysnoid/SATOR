"""Plain-language context for a proposed recipe.

Each proposal reports the predicted mean and standard deviation, how far it
sits from the nearest tested recipe, and which inputs the GP is most
sensitive to. Sensitivity is the inverse lengthscale, mapped back to the
original columns when the GP was trained in PCA or ILR coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mixture import _contrast
from .utils import MAX_GOALS, MIN_GOALS


def _num(value: float) -> str:
    return f"{float(value):.4g}"


def _lengthscales(submodel: Any) -> np.ndarray | None:
    kernel = getattr(submodel, "covar_module", None)
    if kernel is None:
        return None
    base = getattr(kernel, "base_kernel", kernel)
    raw = getattr(base, "lengthscale", None)
    if raw is None:
        return None
    values = np.asarray(raw.detach().cpu().numpy(), dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return values


def nearest_tested(
    recipe: np.ndarray,
    trained: np.ndarray,
    bounds: list[tuple[float, float]],
) -> dict[str, Any]:
    """Euclidean distance in the unit search box. A categorical mismatch counts as 1."""
    row = np.asarray(recipe, dtype=float).reshape(-1)
    data = np.asarray(trained, dtype=float)
    span = np.array([max(float(hi) - float(lo), 1e-9) for lo, hi in bounds], dtype=float)
    distance = np.linalg.norm((data - row) / span, axis=1)
    index = int(np.argmin(distance))
    return {"index": index, "distance": float(distance[index])}


def _input_scores(lengthscale: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(lengthscale, 1e-6)


def _pca_scores(lengthscale: np.ndarray, components: np.ndarray) -> np.ndarray:
    loadings = np.abs(np.asarray(components, dtype=float))
    weights = np.maximum(np.asarray(lengthscale, dtype=float).reshape(-1), 1e-6)
    rank = min(loadings.shape[0], weights.shape[0])
    return (loadings[:rank] / weights[:rank, None]).sum(axis=0)


def _ilr_scores(lengthscale: np.ndarray, encoder: Any) -> np.ndarray:
    scores = np.zeros(encoder.dim, dtype=float)
    weights = np.maximum(np.asarray(lengthscale, dtype=float).reshape(-1), 1e-6)
    cursor = 0
    emitted: set[int] = set()
    for index in range(encoder.dim):
        if index in emitted:
            continue
        group = next((grp for grp in encoder.groups if index in grp), None)
        if group is None:
            scores[index] = 1.0 / weights[cursor]
            cursor += 1
            emitted.add(index)
            continue
        contrast = np.abs(_contrast(len(group)))
        local = weights[cursor : cursor + contrast.shape[0]]
        for column, target in enumerate(group):
            scores[target] = float(np.sum(contrast[:, column] / local))
        cursor += int(contrast.shape[0])
        emitted.update(group)
    return scores


def parameter_sensitivity(
    model: Any,
    names: list[str],
    *,
    n_obj: int,
    space: str,
    components: np.ndarray | None = None,
    encoder: Any = None,
) -> dict[str, Any]:
    """Rank inputs by inverse lengthscale for each objective GP."""
    by_objective: dict[str, list[dict[str, Any]]] = {}
    accumulated: list[np.ndarray] = []
    submodels = list(getattr(model, "models", [model]))
    for index, (name, submodel) in enumerate(zip(names, submodels, strict=False)):
        if index >= n_obj:
            break
        lengthscale = _lengthscales(submodel)
        if lengthscale is None:
            continue
        if space == "pca" and components is not None:
            scores = _pca_scores(lengthscale, components)
            per_input = None
        elif space == "ilr" and encoder is not None:
            scores = _ilr_scores(lengthscale, encoder)
            per_input = None
        elif lengthscale.shape[0] == len(names):
            scores = _input_scores(lengthscale)
            per_input = lengthscale
        else:
            continue
        if scores.shape[0] != len(names):
            continue
        accumulated.append(scores)
        order = np.argsort(-scores)
        rows = []
        for rank in order:
            item: dict[str, Any] = {
                "parameter": names[int(rank)],
                "sensitivity": float(scores[int(rank)]),
            }
            if per_input is not None:
                item["lengthscale"] = float(per_input[int(rank)])
            rows.append(item)
        by_objective[name] = rows
    report: dict[str, Any] = {"space": space, "by_objective": by_objective}
    if accumulated:
        mean_scores = np.mean(np.vstack(accumulated), axis=0)
        order = np.argsort(-mean_scores)
        report["parameters"] = [
            {"parameter": names[int(rank)], "sensitivity": float(mean_scores[int(rank)])} for rank in order
        ]
    else:
        report["parameters"] = []
    return report


def _best_tested(values: np.ndarray, goal: str) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or goal not in MAX_GOALS | MIN_GOALS:
        return None
    return float(np.min(finite) if goal in MIN_GOALS else np.max(finite))


def _improves(mean: float, best: float | None, goal: str) -> bool | None:
    if best is None:
        return None
    if goal in MIN_GOALS:
        return bool(mean < best)
    if goal in MAX_GOALS:
        return bool(mean > best)
    return None


def explain_prediction(
    *,
    means: list[float],
    variances: list[float],
    recipe: np.ndarray,
    trained_x: np.ndarray,
    trained_y: np.ndarray,
    bounds: list[tuple[float, float]],
    objective_names: list[str],
    goals: list[str],
    sensitive: list[str],
) -> dict[str, Any]:
    nearest = nearest_tested(recipe, trained_x, bounds)
    nearest_y = trained_y[nearest["index"]]
    objectives = []
    clauses = []
    improvements: list[bool | None] = []
    for index, name in enumerate(objective_names):
        mean = float(means[index])
        sd = float(np.sqrt(max(variances[index], 0.0)))
        goal = goals[index] if index < len(goals) else "min"
        column = trained_y[:, index] if trained_y.ndim > 1 else trained_y
        best = _best_tested(np.asarray(column, dtype=float), goal)
        better = _improves(mean, best, goal)
        improvements.append(better)
        objectives.append(
            {
                "name": name,
                "mean": mean,
                "sd": sd,
                "best_tested": best,
                "better_than_best": better,
            }
        )
        if better is True:
            clauses.append(f"{name} {_num(mean)} ± {_num(sd)}, improving on the best tested {_num(best)}")
        elif better is False and best is not None:
            clauses.append(f"{name} {_num(mean)} ± {_num(sd)}, short of the best tested {_num(best)}")
        else:
            clauses.append(f"{name} {_num(mean)} ± {_num(sd)}")
    distance = float(nearest["distance"])
    novel = distance > 0.35
    if len(objective_names) > 1:
        role = "explore" if novel else "tradeoff"
    elif improvements and improvements[0] is True and not novel:
        role = "exploit"
    else:
        role = "explore"
    if novel:
        place = (
            f"The nearest tested recipe is {_num(distance)} search-box lengths away, so this step leaves the tested set"
        )
    else:
        place = f"The nearest tested recipe is {_num(distance)} search-box lengths away"
    drivers = ""
    if sensitive:
        shown = ", ".join(sensitive[:3])
        drivers = f". The prediction moves most when {shown} changes"
    summary = f"Predicted {', '.join(clauses)}. {place}{drivers}."
    return {
        "objectives": objectives,
        "nearest_tested": {
            "index": nearest["index"],
            "distance": distance,
            "objectives": [float(v) for v in np.asarray(nearest_y, dtype=float).reshape(-1)],
        },
        "role": role,
        "sensitive_parameters": sensitive[:3],
        "summary": summary,
    }
