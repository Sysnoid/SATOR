"""Replicate wells and a control well that repeats the best recipe so far."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def best_observed_index(y: np.ndarray, signs: np.ndarray) -> int:
    """Row the trust-region replay would centre on. Larger is better after ``signs``."""
    signed = np.atleast_2d(np.asarray(y, dtype=float)) * np.asarray(signs, dtype=float).reshape(1, -1)
    if signed.shape[1] > 1 and np.isfinite(signed).all():
        from .trust_region import _hypervolume

        finite = signed
        ref = finite.min(axis=0) - 0.1 * np.maximum(finite.max(axis=0) - finite.min(axis=0), 1e-6)
        best = 0
        best_value = _hypervolume(signed[:1], ref)
        for index in range(1, signed.shape[0]):
            value = _hypervolume(signed[: index + 1], ref)
            if value > best_value + 1e-12:
                best_value = value
                best = index
        return best
    if signed.shape[1] == 1:
        column = np.where(np.isfinite(signed[:, 0]), signed[:, 0], -np.inf)
        return int(np.argmax(column))
    score = np.zeros(signed.shape[0], dtype=float)
    usable = np.zeros(signed.shape[0], dtype=bool)
    for column in range(signed.shape[1]):
        finite = np.isfinite(signed[:, column])
        score = np.where(finite, score + signed[:, column], score)
        usable |= finite
    return int(np.argmax(np.where(usable, score, -np.inf)))


def posterior_in_user_units(model: Any, x_model: Any, signs: np.ndarray, n_obj: int) -> tuple[list[float], list[float]]:
    """Posterior mean in user units and variance at one model-space row."""
    means: list[float] = []
    variances: list[float] = []
    for index, submodel in enumerate(model.models):
        if index >= n_obj:
            break
        post = submodel.posterior(x_model)
        mean = float(post.mean.detach().cpu().numpy().ravel()[0])
        variance = float(post.variance.detach().cpu().numpy().ravel()[0])
        means.append(mean * float(signs[index]))
        variances.append(variance)
    return means, variances


def expand_replicates(predictions: list[dict[str, Any]], n_replicates: int) -> list[dict[str, Any]]:
    """Repeat each new recipe. A control well is kept as a single copy."""
    copies = max(1, int(n_replicates))
    if copies == 1:
        for pred in predictions:
            pred.setdefault("replicate", 1)
        return predictions
    expanded: list[dict[str, Any]] = []
    for pred in predictions:
        count = 1 if pred.get("control") else copies
        for replicate in range(1, count + 1):
            item = copy.deepcopy(pred)
            item["replicate"] = replicate
            expanded.append(item)
    return expanded
