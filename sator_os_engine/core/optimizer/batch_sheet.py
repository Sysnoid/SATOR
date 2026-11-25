"""Turn proposed fractions into a weighable batch sheet."""

from __future__ import annotations

from typing import Any


def _ingredient_indices(params: list[dict[str, Any]], groups: list[dict[str, Any]]) -> list[int]:
    if groups:
        found = sorted({int(i) for group in groups for i in group.get("indices", [])})
        return [i for i in found if 0 <= i < len(params)]
    return [
        i
        for i, param in enumerate(params)
        if param.get("type", "float") == "float" and float(param.get("min", 0.0)) >= 0.0
    ]


def batch_sheet(
    predictions: list[dict[str, Any]],
    params: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    mass: float,
    unit: str = "g",
) -> dict[str, Any]:
    """Grams per well and per ingredient for a target batch mass.

    Fractions are scaled so the weighed ingredients in each well add up to ``mass``.
    """
    indices = _ingredient_indices(params, groups)
    names = [str(params[i]["name"]) for i in indices]
    totals = {name: 0.0 for name in names}
    rows: list[dict[str, Any]] = []
    for well, pred in enumerate(predictions, start=1):
        candidate = pred.get("candidate") or {}
        raw = [float(candidate.get(name, 0.0)) for name in names]
        scale = sum(raw)
        grams = {}
        if scale > 0.0:
            for name, value in zip(names, raw, strict=True):
                weighed = float(mass) * value / scale
                grams[name] = weighed
                totals[name] += weighed
        else:
            grams = {name: 0.0 for name in names}
        rows.append({"well": well, "grams": grams, "total": float(sum(grams.values()))})
    return {"mass": float(mass), "unit": unit, "rows": rows, "totals": totals}
