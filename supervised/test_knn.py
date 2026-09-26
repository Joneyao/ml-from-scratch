import numpy as np
from supervised.knn import KNN
from common.datasets import load_iris_binary, make_blobs_2d


def test_k1_self_predict_perfect():
    """k=1 时，用训练集自预测应完全正确（准确率 == 1.0）。

    最近邻在 k=1 时对每个训练点找到的最近点就是它自己（距离 0），
    因此预测标签必然等于真实标签。
    """
    X, y = make_blobs_2d(n=100, seed=0)
    clf = KNN(k=1)
    clf.fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc == 1.0


def test_iris_accuracy_above_threshold():
    """在 load_iris_binary 上训练集准确率应高于 0.9。"""
    X, y = load_iris_binary()
    clf = KNN(k=5)
    clf.fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.9


def test_accuracy_matches_sklearn():
    """与 sklearn 的 KNeighborsClassifier 同 k 值对照，准确率差异应小于 0.05。"""
    from sklearn.neighbors import KNeighborsClassifier

    X, y = load_iris_binary()
    k = 5

    clf = KNN(k=k)
    clf.fit(X, y)
    acc_ours = np.mean(clf.predict(X) == y)

    sk = KNeighborsClassifier(n_neighbors=k)
    sk.fit(X, y)
    acc_sk = np.mean(sk.predict(X) == y)

    assert abs(acc_ours - acc_sk) < 0.05
