from __future__ import annotations

from typing import Any

import numpy as np


def enforce_sum_to_target_training(X: np.ndarray, sums_cfg: list[dict[str, Any]] | None) -> np.ndarray:
    if not sums_cfg:
        return X
    Xn = np.asarray(X, dtype=float).copy()
    for sc in sums_cfg:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < Xn.shape[1]]
        if not idxs:
            continue
        target = float(sc.get("target_sum", 1.0))
        sub = Xn[:, idxs]
        totals = np.sum(sub, axis=1, keepdims=True)
        totals_safe = np.where(totals == 0.0, 1.0, totals)
        sub = sub * (target / totals_safe)
        Xn[:, idxs] = sub
    return Xn


class ScaledPCA:
    """PCA on *z-score scaled* features with a raw-X facing API.

    Without scaling, features with large raw variance (e.g. a process-parameter
    range of 5..25) dominate the principal components relative to mass
    fractions in 0..1. That hides recipe structure behind process-knob noise.

    This wrapper fits ``StandardScaler`` then sklearn ``PCA`` on the scaled
    data, but exposes the same surface as a plain sklearn PCA so callers can
    keep using ``transform`` / ``inverse_transform`` / ``.components_`` /
    ``.mean_`` unchanged. In particular:

    * ``transform(X)`` and ``fit_transform(X)`` take **raw** ``X`` and return the
      PCA coordinates of ``scale(X)`` (what the GP sees).
    * ``inverse_transform(Z)`` returns **raw** ``X`` (unscaled).
    * ``components_`` and ``mean_`` are the *effective* encoder such that
      ``(x - mean_) @ components_.T == (scale(x) - pca_mean) @ pca_components.T``.
      This lets the SLSQP reconstructor compute encoded coordinates directly
      from raw ``x`` without a separate scaler.

    The inverse identity ``Z @ components_ + mean_`` does NOT yield raw ``X``
    (it requires per-feature ``sigma`` multiplication); always call
    ``inverse_transform`` for the unscaling path.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = int(n_components)
        self._pca = None
        self._mu_s: np.ndarray | None = None
        self._sigma_s: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_scale_: np.ndarray | None = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._mu_s) / self._sigma_s

    def _unscale(self, X_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(X_scaled, dtype=float) * self._sigma_s + self._mu_s

    def fit(self, X: np.ndarray) -> ScaledPCA:
        from sklearn.decomposition import PCA

        X = np.asarray(X, dtype=float)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        # Guard against zero-variance features; leaving them at ``1`` keeps the
        # (x - mu)/sigma operation well-defined and the zero column cancels in
        # the PCA fit.
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        self._mu_s = mu
        self._sigma_s = sigma

        X_scaled = (X - mu) / sigma
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(X_scaled)

        C = self._pca.components_
        mu_p = self._pca.mean_
        self.components_ = C / sigma[None, :]
        self.mean_ = mu + sigma * mu_p
        self.explained_variance_ = self._pca.explained_variance_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.scaler_mean_ = mu
        self.scaler_scale_ = sigma
        self.k_requested_: int | str | None = None
        self.k_effective_: int | None = None
        self.k_dropped_: int = 0
        return self

    def retain_components(self, keep: np.ndarray) -> None:
        """Drop principal components whose ``keep`` entry is false."""
        if self._pca is None or self.components_ is None:
            raise RuntimeError("call fit() first")
        mask = np.asarray(keep, dtype=bool)
        if mask.shape[0] != self.components_.shape[0]:
            raise ValueError("keep mask does not match the number of components")
        if not np.any(mask):
            raise ValueError("PCA components are degenerate (no component passed the variance floor)")
        self._pca.components_ = self._pca.components_[mask]
        self._pca.explained_variance_ = self._pca.explained_variance_[mask]
        self._pca.explained_variance_ratio_ = self._pca.explained_variance_ratio_[mask]
        if getattr(self._pca, "singular_values_", None) is not None:
            self._pca.singular_values_ = self._pca.singular_values_[mask]
        self._pca.n_components_ = int(np.sum(mask))
        self.n_components = int(np.sum(mask))
        self.components_ = self.components_[mask]
        self.explained_variance_ = np.asarray(self._pca.explained_variance_)
        self.explained_variance_ratio_ = np.asarray(self._pca.explained_variance_ratio_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._pca is not None, "call fit() first"
        return self._pca.transform(self._scale(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        assert self._pca is not None, "call fit() first"
        X_scaled = self._pca.inverse_transform(np.asarray(Z, dtype=float))
        return self._unscale(X_scaled)


def _pca_component_limit(n_rows: int, n_features: int) -> int:
    limit = min(n_rows - 1, n_features)
    if limit < 1:
        raise ValueError(f"pca_dimension requires at least 2 rows and 1 column; got n={n_rows}, d={n_features}")
    return limit


def resolve_pca_k(n_rows: int, n_features: int, k: int | str) -> tuple[int, bool]:
    """Return ``(k_fit, auto)`` or raise when ``k`` is outside ``1..min(n-1, d)``."""
    limit = _pca_component_limit(n_rows, n_features)
    if isinstance(k, str) and str(k).strip().lower() == "auto":
        return min(limit, 10), True
    try:
        ki = int(k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pca_dimension must be an integer or 'auto' in 1..{limit}; got {k!r}") from exc
    if ki < 1 or ki > limit:
        raise ValueError(f"pca_dimension must satisfy 1 <= k <= min(n-1, d)={limit}; got {ki}")
    return ki, False


def fit_pca_normalize(X: np.ndarray, k: int | str):
    """Fit ``ScaledPCA`` and return ``(pca, pc_mins, pc_maxs, pc_range, Z_norm)``.

    ``k`` must satisfy ``1 <= k <= min(n-1, d)``. ``k="auto"`` keeps the smallest
    prefix with at least 95% explained variance, capped at ``min(n-1, d, 10)``.
    Components with explained-variance ratio below ``1e-10`` or a score range
    below ``1e-9`` are dropped; ``pca.k_effective_`` reports how many remain.
    """
    x = np.asarray(X, dtype=float)
    n_rows, n_features = x.shape
    k_fit, auto = resolve_pca_k(n_rows, n_features, k)
    pca = ScaledPCA(n_components=k_fit)
    z_raw = pca.fit_transform(x)
    ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    keep = np.ones(k_fit, dtype=bool)
    if auto:
        cumulative = np.cumsum(np.nan_to_num(ratio, nan=0.0))
        k_auto = int(np.searchsorted(cumulative, 0.95) + 1)
        k_auto = max(1, min(k_auto, k_fit))
        keep[:] = False
        keep[:k_auto] = True
    pc_range_raw = np.max(z_raw, axis=0) - np.min(z_raw, axis=0)
    keep = keep & np.isfinite(ratio) & (ratio >= 1e-10) & (pc_range_raw >= 1e-9)
    n_before = int(k_fit)
    if not np.all(keep):
        pca.retain_components(keep)
        z_raw = pca.transform(x)
    pc_mins = np.min(z_raw, axis=0)
    pc_maxs = np.max(z_raw, axis=0)
    pc_range = np.maximum(pc_maxs - pc_mins, 1e-12)
    z_norm = (z_raw - pc_mins) / pc_range
    pca.k_requested_ = "auto" if auto else int(k)
    pca.k_effective_ = int(z_norm.shape[1])
    pca.k_dropped_ = int(n_before - pca.k_effective_)
    return pca, pc_mins, pc_maxs, pc_range, z_norm


def z_norm_to_input(pca, pc_mins: np.ndarray, pc_range: np.ndarray, z_norm: np.ndarray) -> np.ndarray:
    z_raw = z_norm * pc_range + pc_mins
    return pca.inverse_transform(z_raw)


def input_to_z_norm(pca, pc_mins: np.ndarray, pc_range: np.ndarray, X: np.ndarray) -> np.ndarray:
    z_raw = pca.transform(X)
    return (z_raw - pc_mins) / pc_range
