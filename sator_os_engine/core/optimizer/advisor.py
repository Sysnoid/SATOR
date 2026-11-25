"""Choose a surrogate and a search route from the shape of the history.

This is a rule, not a cross-validated score. The reason is returned so a caller
can override it. PCA is left alone when the request asked for it.
"""

from __future__ import annotations


def advise_route(
    n_rows: int,
    n_inputs: int,
    *,
    model: str,
    use_pca: bool,
    mode: str,
    categorical: bool,
    ilr: bool,
) -> dict[str, str]:
    """Return ``model``, ``route``, and a one-sentence ``reason``."""
    if use_pca:
        return {
            "model": model,
            "route": "direct",
            "reason": "PCA was requested, so the advisor leaves the model and the route unchanged.",
        }
    long_history = n_rows >= 300 or (mode == "throughput" and n_rows >= 150 and n_inputs >= 10)
    if long_history:
        return {
            "model": model,
            "route": "trust_region",
            "reason": "The history is long enough to search a trust region around the best recipe instead of the whole box.",
        }
    sparse = model == "gp" and n_inputs >= 20 and n_rows <= 200 and not categorical and not ilr
    if sparse:
        return {
            "model": "saas_map",
            "route": "direct",
            "reason": "Many inputs and a short history: a sparse prior keeps unsupported ingredients from dominating the GP.",
        }
    return {
        "model": model,
        "route": "direct",
        "reason": "A standard GP on the original inputs matches this number of rows and inputs.",
    }
