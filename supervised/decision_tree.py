"""决策树（分类）——从零手写基尼系数 + 递归分裂 + 预剪枝。

决策树的核心是不断问"是非题"把样本切开：每次在所有特征、所有候选
阈值里，挑一个让子集"最纯"的切法。衡量纯度用基尼不纯度（Gini
impurity）——一组标签全是同一类时基尼为 0，越混乱基尼越大。选分裂
点时，比较分裂前后的加权基尼，下降最多的那一刀就是最佳分裂。

建树是递归的：切一刀得到左右两个子集，再对子集各自递归地切，直到
碰到停止条件——纯度已经为 0、深度到达 max_depth、或样本数少于
min_samples_split。这两个上限就是预剪枝，防止树长得太深过拟合。
到达停止条件的节点变成叶节点，存下该子集里的多数类作为预测结果。
整篇只用 NumPy。
"""
import numpy as np


def gini(y):
    """计算一组标签的基尼不纯度。

    定义为 1 - Σ p_k^2，其中 p_k 是第 k 类在 y 中的占比。全是同一
    类时 Σ p_k^2 = 1，基尼为 0（最纯）；二分类各占一半时基尼取最大
    值 0.5。

    参数：
        y: 标签数组，形状 (n_samples,)。

    返回：
        基尼不纯度（float）。空数组约定返回 0.0。
    """
    y = np.asarray(y)
    n = y.shape[0]
    if n == 0:
        return 0.0
    # 每个类别的出现次数 → 占比
    counts = np.bincount(y)
    probs = counts / n
    return 1.0 - np.sum(probs * probs)


class _Node:
    """决策树的一个节点。

    内部节点存分裂用的特征下标、阈值和左右子树；叶节点只存预测标签
    （多数类），此时 feature 为 None。
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature      # 分裂特征下标
        self.threshold = threshold  # 分裂阈值：<= 阈值走左，> 阈值走右
        self.left = left            # 左子树
        self.right = right          # 右子树
        self.label = label          # 叶节点的多数类标签

    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    """决策树分类器（CART，基尼系数），纯 NumPy 实现。

    参数：
        max_depth:         树的最大深度（预剪枝），到达后强制变叶节点。
        min_samples_split: 节点继续分裂所需的最少样本数（预剪枝），
                           样本数不足时变叶节点。

    属性：
        root_: 建好的树根节点。
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_ = None

    def fit(self, X, y):
        """递归建树。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。
            y: 标签，形状 (n_samples,)，须为非负整数类别。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root_ = self._build(X, y, depth=0)
        return self

    def _majority(self, y):
        """返回 y 中出现次数最多的类别（平票取最小标签，行为确定）。"""
        return int(np.bincount(y).argmax())

    def _best_split(self, X, y):
        """遍历所有特征和候选阈值，找基尼下降最大的最佳分裂。

        对每个特征，把该列出现过的值排序，取相邻值的中点作为候选阈值
        （只在类别可能改变的边界处切，避免无意义的重复切分）。用分裂
        后左右子集的加权基尼衡量分裂质量，越小越好。

        返回：
            (best_feature, best_threshold)；若找不到能降低基尼的分裂，
            返回 (None, None)。
        """
        n_samples, n_features = X.shape
        parent_gini = gini(y)
        best_gain = 0.0
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            values = X[:, feature]
            uniq = np.unique(values)
            if uniq.shape[0] < 2:
                continue
            # 相邻唯一值的中点作为候选阈值
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for thr in thresholds:
                left_mask = values <= thr
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                # 分裂后的加权基尼
                weighted = (n_left * gini(y[left_mask]) +
                            n_right * gini(y[right_mask])) / n_samples
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = thr

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        """递归构建子树，返回节点。"""
        # 预剪枝 / 停止条件：纯节点、深度到顶、样本太少 → 叶节点
        if (gini(y) == 0.0 or
                depth >= self.max_depth or
                y.shape[0] < self.min_samples_split):
            return _Node(label=self._majority(y))

        feature, threshold = self._best_split(X, y)
        # 找不到有效分裂（无法进一步降低基尼）→ 叶节点
        if feature is None:
            return _Node(label=self._majority(y))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _predict_one(self, x, node):
        """从根节点开始，按分裂条件一路走到叶节点，返回其标签。"""
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, X):
        """对每个样本沿树下行到叶节点，返回叶节点的多数类。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            预测标签数组，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])
