import numpy as np
import pytest
from fastridge import RidgeEM, RidgeLOOCV


def _data(seed=0, n=200, p=5):
    rng = np.random.default_rng(seed)
    beta = np.array([1., -2., 0.5, 3., -1.5])
    X = rng.standard_normal((n, p))
    y = X @ beta + 0.05 * rng.standard_normal(n)
    return X, y, beta


def test_ridge_em_1d_reconstruction():
    X, y, beta = _data()
    est = RidgeEM().fit(X, y)
    np.testing.assert_allclose(est.coef_, beta, atol=0.1)
    assert est.coef_.ndim == 1
    assert np.isscalar(est.sigma_square_) or est.sigma_square_.ndim == 0
    assert est.iterations_ > 0
    assert est.svd_time_ >= 0


def test_ridge_em_trace_space_original():
    X, y, _ = _data()
    est = RidgeEM(trace=True, trace_space='original').fit(X, y)
    np.testing.assert_allclose(est.coefs_[-1], est.coef_, rtol=1e-10)


def test_ridge_em_trace_space_projected():
    X, y, _ = _data()
    est = RidgeEM(trace=True, trace_space='projected').fit(X, y)
    assert est.coefs_[-1].shape == (5,)
    assert not np.allclose(est.coefs_[-1], est.coef_)


def test_ridge_em_trace_length():
    X, y, _ = _data()
    est = RidgeEM(trace=True).fit(X, y)
    assert len(est.coefs_) == est.iterations_ + 1
    assert len(est.sigma_squares_) == est.iterations_ + 1
    assert len(est.tau_squares_) == est.iterations_ + 1


def test_ridge_em_2d_coef_shape():
    X, y, _ = _data()
    Y = np.column_stack([y, -y, 0.5 * y])
    est = RidgeEM().fit(X, Y)
    assert est.coef_.shape == (3, 5)
    assert est.alpha_.shape == (3,)
    assert est.sigma_square_.shape == (3,)


def test_ridge_em_2d_single_target_not_squeezed():
    X, y, _ = _data()
    est = RidgeEM().fit(X, y[:, None])
    assert est.coef_.shape == (1, 5)
    assert est.alpha_.shape == (1,)


def test_ridge_em_2d_column_matches_1d():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est_2d = RidgeEM().fit(X, Y)
    est_1d = RidgeEM().fit(X, y)
    np.testing.assert_allclose(est_2d.coef_[0], est_1d.coef_, rtol=1e-10)


def test_ridge_em_2d_trace():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est = RidgeEM(trace=True).fit(X, Y)
    assert len(est.sigma_squares_) == 2
    assert len(est.tau_squares_) == 2
    assert len(est.coefs_) == 2


def test_ridge_em_predict_1d():
    X, y, _ = _data()
    est = RidgeEM().fit(X, y)
    assert est.predict(X).shape == (200,)


def test_ridge_em_predict_2d():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est = RidgeEM().fit(X, Y)
    assert est.predict(X).shape == (200, 2)


def test_ridge_loocv_2d_coef_shape():
    X, y, _ = _data()
    Y = np.column_stack([y, -y, 0.5 * y])
    est = RidgeLOOCV().fit(X, Y)
    assert est.coef_.shape == (3, 5)
    assert est.alpha_.shape == (3,)
    assert est.sigma_square_.shape == (3,)


def test_ridge_loocv_2d_single_target_not_squeezed():
    X, y, _ = _data()
    est = RidgeLOOCV().fit(X, y[:, None])
    assert est.coef_.shape == (1, 5)
    assert est.alpha_.shape == (1,)


def test_ridge_loocv_2d_column_matches_1d():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est_2d = RidgeLOOCV().fit(X, Y)
    est_1d = RidgeLOOCV().fit(X, y)
    np.testing.assert_allclose(est_2d.coef_[0], est_1d.coef_, rtol=1e-10)
    np.testing.assert_allclose(est_2d.alpha_[0], est_1d.alpha_, rtol=1e-10)


def test_ridge_loocv_predict_1d():
    X, y, _ = _data()
    est = RidgeLOOCV().fit(X, y)
    assert est.predict(X).shape == (200,)


def test_ridge_loocv_predict_2d():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est = RidgeLOOCV().fit(X, Y)
    assert est.predict(X).shape == (200, 2)
