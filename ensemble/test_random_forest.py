import numpy as np
from ensemble.random_forest import RandomForest
from supervised.decision_tree import DecisionTree
from common.datasets import load_breast_cancer_data


def _train_test_split(X, y, test_ratio=0.3, seed=0):
    """简单留出法：按比例把数据切成训练集和测试集。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def test_fit_predict_shape():
    """森林能 fit/predict，预测输出形状与输入样本数一致。"""
    X, y = load_breast_cancer_data()
    clf = RandomForest(n_trees=10, max_depth=5, seed=0)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (X.shape[0],)
    # 预测标签应落在训练时见过的类别集合内
    assert set(np.unique(pred)).issubset(set(np.unique(y)))


def test_forest_not_worse_than_single_tree():
    """在乳腺癌数据上，随机森林平均准确率应不差于单棵决策树，体现集成价值。

    集成"不差于单树"是统计意义上的期望结论，而非每一次随机划分都成立
    ——在某个单独的留出划分上，浅树偶尔可能打平甚至略胜森林。因此这里
    在多个不同划分上取平均准确率来对照，反映集成的稳定优势。
    """
    X, y = load_breast_cancer_data()

    forest_accs, tree_accs = [], []
    for seed in range(5):
        X_tr, X_te, y_tr, y_te = _train_test_split(X, y, test_ratio=0.3, seed=seed)

        forest = RandomForest(n_trees=15, max_depth=5, seed=0)
        forest.fit(X_tr, y_tr)
        forest_accs.append(np.mean(forest.predict(X_te) == y_te))

        tree = DecisionTree(max_depth=5)
        tree.fit(X_tr, y_tr)
        tree_accs.append(np.mean(tree.predict(X_te) == y_te))

    assert np.mean(forest_accs) >= np.mean(tree_accs)


def test_accuracy_matches_sklearn():
    """与 sklearn 的 RandomForestClassifier 对照，测试集准确率差异应小于 0.1。"""
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer_data()
    X_tr, X_te, y_tr, y_te = _train_test_split(X, y, test_ratio=0.3, seed=7)

    ours = RandomForest(n_trees=20, max_depth=5, seed=0)
    ours.fit(X_tr, y_tr)
    acc_ours = np.mean(ours.predict(X_te) == y_te)

    sk = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0)
    sk.fit(X_tr, y_tr)
    acc_sk = np.mean(sk.predict(X_te) == y_te)

    assert abs(acc_ours - acc_sk) < 0.1
