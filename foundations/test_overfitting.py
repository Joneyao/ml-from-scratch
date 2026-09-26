import numpy as np
from foundations.overfitting import poly_features, train_test_errors


def test_poly_features_shape():
    X = np.array([1.0, 2.0, 3.0])
    P = poly_features(X, degree=3)
    assert P.shape == (3, 3)
    assert np.allclose(P[:, 0], X)        # 一次项
    assert np.allclose(P[:, 2], X ** 3)   # 三次项


def test_high_degree_overfits_without_reg():
    # 高阶无正则：训练误差很低，测试误差明显更高
    tr, te = train_test_errors(degree=12, alpha=0.0, seed=0)
    assert tr < te
    assert te > tr * 2


def test_regularization_reduces_test_error():
    _, te_no_reg = train_test_errors(degree=12, alpha=0.0, seed=0)
    _, te_reg = train_test_errors(degree=12, alpha=1.0, seed=0)
    assert te_reg < te_no_reg
