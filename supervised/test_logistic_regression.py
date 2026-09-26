import numpy as np
from supervised.logistic_regression import LogisticRegression, sigmoid
from common.datasets import make_blobs_2d


def test_sigmoid_at_zero():
    """sigmoid(0) 应等于 0.5。"""
    assert abs(sigmoid(0.0) - 0.5) < 1e-12


def test_sigmoid_stable_on_extremes():
    """极端输入不应溢出，且落在 (0, 1) 内。"""
    assert 0.0 <= sigmoid(-1000.0) < 1e-6
    assert 1.0 - 1e-6 < sigmoid(1000.0) <= 1.0


def test_fit_accuracy_above_threshold():
    """在两簇数据上训练后，训练集准确率应高于 0.9。"""
    X, y = make_blobs_2d(n=200, seed=0)
    clf = LogisticRegression()
    clf.fit(X, y, lr=0.1, n_iters=2000)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.9


def test_predict_proba_range():
    """predict_proba 输出应落在 [0, 1] 区间。"""
    X, y = make_blobs_2d(n=100, seed=1)
    clf = LogisticRegression()
    clf.fit(X, y, lr=0.1, n_iters=1000)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0],)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_accuracy_matches_sklearn():
    """与 sklearn 的逻辑回归在同一数据上的准确率对照，差异应小于 0.05。"""
    from sklearn.linear_model import LogisticRegression as SkLR

    X, y = make_blobs_2d(n=200, seed=2)

    clf = LogisticRegression()
    clf.fit(X, y, lr=0.1, n_iters=3000)
    acc_ours = np.mean(clf.predict(X) == y)

    sk = SkLR()
    sk.fit(X, y)
    acc_sk = np.mean(sk.predict(X) == y)

    assert abs(acc_ours - acc_sk) < 0.05
