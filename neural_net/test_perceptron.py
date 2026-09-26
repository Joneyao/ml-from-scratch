import numpy as np
from neural_net.perceptron import Perceptron, xor_data
from common.datasets import make_blobs_2d


def test_linearly_separable_accuracy():
    """线性可分的两簇数据上，感知机训练后准确率应高于 0.95。

    seed=1 生成的两簇几乎不重叠（线性可分），感知机学习规则能收敛到
    一条完美分开两类的直线。注意 seed=0 的两簇彼此重叠、线性不可分，
    那种情况下任何线性分类器（含 sklearn）都到不了 0.95。
    """
    X, y = make_blobs_2d(n=200, seed=1)
    clf = Perceptron(lr=0.1, n_iters=100)
    clf.fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.95


def test_xor_is_unlearnable():
    """单个感知机学不会异或（XOR），准确率明显低于 1.0。"""
    X, y = xor_data()
    clf = Perceptron(lr=0.1, n_iters=1000)
    clf.fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    # XOR 线性不可分，单层感知机最多只能分对 3/4 的点
    assert acc <= 0.75


def test_accuracy_matches_sklearn():
    """与 sklearn 感知机在线性可分数据上的准确率对照，差异应小于 0.1。"""
    from sklearn.linear_model import Perceptron as SkPerceptron

    X, y = make_blobs_2d(n=200, seed=2)

    clf = Perceptron(lr=0.1, n_iters=100)
    clf.fit(X, y)
    acc_ours = np.mean(clf.predict(X) == y)

    sk = SkPerceptron()
    sk.fit(X, y)
    acc_sk = np.mean(sk.predict(X) == y)

    assert abs(acc_ours - acc_sk) < 0.1
