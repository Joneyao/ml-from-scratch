import numpy as np
from supervised.decision_tree import DecisionTree, gini
from common.datasets import load_iris_binary


def test_gini_pure_vs_mixed():
    """单一类别的基尼系数应为 0，混合均匀的基尼系数应大于 0。

    基尼不纯度衡量一组标签的"混乱程度"：全是同一类时纯度最高、
    不纯度为 0；两类各占一半时最混乱，二分类下取最大值 0.5。
    """
    pure = np.array([1, 1, 1, 1])
    assert gini(pure) == 0.0

    mixed = np.array([0, 0, 1, 1])
    assert gini(mixed) > 0.0
    # 二分类 50/50 时基尼恰为 0.5
    assert abs(gini(mixed) - 0.5) < 1e-9


def test_iris_accuracy_above_threshold():
    """在 load_iris_binary 上训练集准确率应高于 0.9。"""
    X, y = load_iris_binary()
    clf = DecisionTree(max_depth=5)
    clf.fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.9


def test_accuracy_matches_sklearn():
    """与 sklearn 的 DecisionTreeClassifier 同 max_depth 对照，准确率差异应小于 0.1。"""
    from sklearn.tree import DecisionTreeClassifier

    X, y = load_iris_binary()
    max_depth = 5

    clf = DecisionTree(max_depth=max_depth)
    clf.fit(X, y)
    acc_ours = np.mean(clf.predict(X) == y)

    sk = DecisionTreeClassifier(max_depth=max_depth, criterion="gini", random_state=0)
    sk.fit(X, y)
    acc_sk = np.mean(sk.predict(X) == y)

    assert abs(acc_ours - acc_sk) < 0.1
