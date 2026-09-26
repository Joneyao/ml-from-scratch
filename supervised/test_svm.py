import numpy as np
from supervised.svm import LinearSVM, rbf_kernel
from common.datasets import make_blobs_2d


def test_linear_svm_accuracy_above_threshold():
    """在线性可分的两簇数据上训练后，准确率应高于 0.95。"""
    X, y = make_blobs_2d(n=200, seed=1)
    clf = LinearSVM()
    clf.fit(X, y, lr=0.01, n_iters=2000, C=1.0)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.95


def test_rbf_kernel_shape_and_diagonal():
    """rbf_kernel(X, X) 应是方阵，且对角线元素全为 1（同点距离 0，exp(0)=1）。"""
    X, _ = make_blobs_2d(n=30, seed=1)
    K = rbf_kernel(X, X, gamma=0.5)
    assert K.shape == (X.shape[0], X.shape[0])
    assert np.allclose(np.diag(K), 1.0)


def test_accuracy_matches_sklearn():
    """与 sklearn 的线性 SVM 在同一数据上的准确率对照，差异应小于 0.1。"""
    from sklearn.svm import SVC

    X, y = make_blobs_2d(n=200, seed=2)

    clf = LinearSVM()
    clf.fit(X, y, lr=0.01, n_iters=3000, C=1.0)
    acc_ours = np.mean(clf.predict(X) == y)

    sk = SVC(kernel="linear")
    sk.fit(X, y)
    acc_sk = np.mean(sk.predict(X) == y)

    assert abs(acc_ours - acc_sk) < 0.1


def test_predict_returns_original_labels():
    """predict 输出应是原始 0/1 标签，而非内部的 ±1。"""
    X, y = make_blobs_2d(n=100, seed=3)
    clf = LinearSVM()
    clf.fit(X, y, lr=0.01, n_iters=1000, C=1.0)
    pred = clf.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})
