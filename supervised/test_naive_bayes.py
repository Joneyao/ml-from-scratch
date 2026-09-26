import numpy as np
from supervised.naive_bayes import GaussianNB, MultinomialNB
from common.datasets import load_iris_binary


def test_gaussian_nb_accuracy_above_threshold():
    """GaussianNB 在 iris 二分类数据上训练集准确率应高于 0.9。"""
    X, y = load_iris_binary()
    clf = GaussianNB()
    clf.fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.9


def test_multinomial_nb_classifies_small_counts():
    """MultinomialNB 在手造的小词频矩阵上应把两类分对。

    类 0 偏向前两个词，类 1 偏向后两个词。用训练样本自身
    验证分类结果与标签一致（简单可分场景）。
    """
    # 4 个词、两类各 2 行
    X = np.array([
        [3, 2, 0, 0],   # 类 0：前两个词多
        [4, 1, 0, 1],   # 类 0
        [0, 0, 3, 2],   # 类 1：后两个词多
        [1, 0, 2, 4],   # 类 1
    ])
    y = np.array([0, 0, 1, 1])

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, y)

    # 训练样本应全部分对
    assert np.array_equal(clf.predict(X), y)

    # 一条明显偏向类 1 的新样本
    new = np.array([[0, 1, 4, 5]])
    assert clf.predict(new)[0] == 1


def test_gaussian_nb_matches_sklearn():
    """GaussianNB 与 sklearn.naive_bayes.GaussianNB 准确率对照，差异应小于 0.05。"""
    from sklearn.naive_bayes import GaussianNB as SkGNB

    X, y = load_iris_binary()

    clf = GaussianNB()
    clf.fit(X, y)
    acc_ours = np.mean(clf.predict(X) == y)

    sk = SkGNB()
    sk.fit(X, y)
    acc_sk = np.mean(sk.predict(X) == y)

    assert abs(acc_ours - acc_sk) < 0.05
