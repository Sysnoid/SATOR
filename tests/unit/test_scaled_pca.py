"""Numerical checks for the ``ScaledPCA`` wrapper.

These tests lock in the contract that matters for the downstream SLSQP
reconstruction path: ``(x - mean_) @ components_.T`` applied to raw ``x``
must equal ``PCA`` on the z-score-scaled data, and ``inverse_transform``
must be the true inverse of ``transform`` on raw ``X``.
"""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.optimizer.preprocess import ScaledPCA, fit_pca_normalize


def _make_wide_scale_data(n: int = 60, d: int = 5, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Deliberately mix feature scales: columns 0..2 are small, 3 is big, 4 is tiny.
    scales = np.array([1.0, 0.5, 0.2, 50.0, 0.001])
    offsets = np.array([0.0, 10.0, -3.0, 5.0, 0.0])
    base = rng.normal(size=(n, d))
    return base * scales + offsets


def test_scaled_pca_forward_identity_matches_inner_pca_on_scaled_data():
    X = _make_wide_scale_data()
    pca = ScaledPCA(n_components=2).fit(X)

    scaled = (X - pca.scaler_mean_) / pca.scaler_scale_
    Z_inner = pca._pca.transform(scaled)
    Z_via_effective = (X - pca.mean_) @ pca.components_.T
    np.testing.assert_allclose(Z_inner, Z_via_effective, atol=1e-10, rtol=1e-10)


def test_scaled_pca_transform_inverse_roundtrip_full_rank():
    X = _make_wide_scale_data()
    pca = ScaledPCA(n_components=X.shape[1]).fit(X)
    Z = pca.transform(X)
    X_round = pca.inverse_transform(Z)
    np.testing.assert_allclose(X, X_round, atol=1e-8, rtol=1e-8)


def test_scaled_pca_explains_big_feature_less_than_unscaled_would():
    """With scaling, a single very-high-variance feature must not dominate PC1.

    We compare explained variance ratio on scaled vs raw data and assert the
    scaled PC1 is noticeably lower than the raw PC1 when one feature is 50x
    larger than the others.
    """
    X = _make_wide_scale_data()

    scaled_pca = ScaledPCA(n_components=min(X.shape)).fit(X)
    ev_scaled = scaled_pca.explained_variance_ratio_[0]

    from sklearn.decomposition import PCA

    raw_pca = PCA(n_components=min(X.shape)).fit(X)
    ev_raw = raw_pca.explained_variance_ratio_[0]

    assert ev_raw > 0.99, f"raw PCA should be dominated by big feature (got {ev_raw:.3f})"
    assert ev_scaled < 0.6, f"scaled PC1 should not dominate (got {ev_scaled:.3f})"


def test_fit_pca_normalize_returns_wrapper_and_valid_norm():
    X = _make_wide_scale_data()
    pca, pc_mins, pc_maxs, pc_range, Z_norm = fit_pca_normalize(X, k=2)
    assert isinstance(pca, ScaledPCA)
    assert Z_norm.shape == (X.shape[0], 2)
    assert np.all(Z_norm >= -1e-9) and np.all(Z_norm <= 1 + 1e-9)
    assert np.allclose(pc_range, pc_maxs - pc_mins, atol=1e-12)
