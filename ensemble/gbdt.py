"""GBDT（梯度提升决策树）——从零手写回归树 + 残差拟合 + Boosting。

GBDT 属于 Boosting 家族，和随机森林（Bagging）走的是完全不同的路子：

- Bagging（如随机森林）是"并行投票"：一次性训练很多棵互相独立的树，
  每棵树看不同的样本/特征子集，最后把它们的结果平均或投票。树与树之
  间没有先后依赖，谁也不知道谁的存在，靠"人多力量大"降低方差。
- Boosting（如 GBDT）是"串行纠错"：树一棵接一棵地训练，后一棵专门去
  拟合前面所有树加起来还没拟合好的部分——也就是残差。每加一棵树，整
  体预测就往真值靠近一点，靠"逐步改错"降低偏差。

具体到回归任务，GBDT 的流程是：
1. 初始预测取 y 的均值（最省事的常数预测）。
2. 每一轮先算残差 = 真值 - 当前预测，这就是"当前还差多少"。
3. 训练一棵回归树去拟合这个残差。
4. 把树的预测乘上学习率 learning_rate，累加到当前预测上（学习率控制
   每步纠错的幅度，越小越稳但需要越多树）。
5. 重复 n_estimators 轮，把每棵树都存下来。

预测时，最终结果 = 初始均值 + learning_rate × 各棵树预测之和。

注意 GBDT 的基学习器是**回归树**（不是分类树）：按方差（MSE）下降选
分裂点，叶子输出子集残差的均值。这与项目里 supervised/decision_tree.py
的分类树（按基尼分裂、叶子输出多数类）不同，所以本模块自己实现了一
个简单回归树 _RegressionTree，不复用也不改动那份分类树。整篇只用 NumPy。
"""
import numpy as np


class _RegNode:
    """回归树的一个节点。

    内部节点存分裂特征下标、阈值和左右子树；叶节点存该子集的均值作为
    预测值（value），此时 feature 为 None。
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 分裂特征下标
        self.threshold = threshold  # 分裂阈值：<= 阈值走左，> 阈值走右
        self.left = left            # 左子树
        self.right = right          # 右子树
        self.value = value          # 叶节点的预测值（子集均值）

    def is_leaf(self):
        return self.feature is None


class _RegressionTree:
    """回归树（CART 回归版）——按方差下降分裂，叶子输出均值。纯 NumPy。

    与分类树的区别在于分裂准则和叶子取值：
    - 分裂准则用方差（等价于 MSE）。一组目标值越集中方差越小，分裂时
      挑让左右子集加权方差下降最多的那一刀。
    - 叶子节点输出该子集目标值的均值（对 MSE 而言，均值是最优常数预测）。

    在 GBDT 里，这棵树拟合的目标不是原始 y，而是当前的残差。

    参数：
        max_depth: 树的最大深度（预剪枝），到达后强制变叶节点。

    属性：
        root_: 建好的树根节点。
    """

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root_ = None

    def fit(self, X, residual):
        """递归建树，拟合给定的目标值（在 GBDT 中即残差）。

        参数：
            X:        特征矩阵，形状 (n_samples, n_features)。
            residual: 待拟合的目标值，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        residual = np.asarray(residual, dtype=float)
        self.root_ = self._build(X, residual, depth=0)
        return self

    def _best_split(self, X, y):
        """遍历所有特征和候选阈值，找加权方差下降最大的最佳分裂。

        对每个特征，取相邻唯一值的中点作为候选阈值，用分裂后左右子集
        的加权方差（按样本数加权）衡量分裂质量，越小越好。

        返回：
            (best_feature, best_threshold)；若找不到能降低方差的分裂，
            返回 (None, None)。
        """
        n_samples, n_features = X.shape
        # 父节点的加权方差基准（乘以 n 便于和子集直接比较总平方误差）
        parent_cost = np.var(y) * n_samples
        best_cost = parent_cost
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            values = X[:, feature]
            uniq = np.unique(values)
            if uniq.shape[0] < 2:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for thr in thresholds:
                left_mask = values <= thr
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                # 左右子集的总平方误差之和（方差 × 样本数）
                cost = (np.var(y[left_mask]) * n_left +
                        np.var(y[right_mask]) * n_right)
                if cost < best_cost:
                    best_cost = cost
                    best_feature = feature
                    best_threshold = thr

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        """递归构建子树，返回节点。"""
        # 预剪枝 / 停止条件：目标值已完全一致、深度到顶、样本太少 → 叶节点
        if depth >= self.max_depth or y.shape[0] <= 1 or np.all(y == y[0]):
            return _RegNode(value=float(np.mean(y)))

        feature, threshold = self._best_split(X, y)
        # 找不到有效分裂（无法进一步降低方差）→ 叶节点
        if feature is None:
            return _RegNode(value=float(np.mean(y)))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return _RegNode(feature=feature, threshold=threshold, left=left, right=right)

    def _predict_one(self, x, node):
        """从根节点开始，按分裂条件一路走到叶节点，返回其均值预测。"""
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        """对每个样本沿树下行到叶节点，返回叶节点存的均值。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            预测值数组，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])


class GBDT:
    """梯度提升决策树（回归），纯 NumPy 实现。

    用于回归任务：串行训练多棵回归树，每棵树拟合前面所有树累加后仍
    残留的残差，逐步把整体预测拉向真值。损失用平方误差（MSE），此时
    "负梯度"恰好等于残差 y - 预测，所以每棵树直接拟合残差即可。

    参数：
        n_estimators:  树的数量（Boosting 轮数），越多拟合越强但可能过拟合。
        learning_rate: 学习率，每棵树的贡献缩放系数。越小越稳，通常需要
                       配更多的树；起到正则化作用。
        max_depth:     单棵回归树的最大深度，控制每棵树的复杂度。

    属性：
        init_:        初始预测（训练集 y 的均值）。
        trees_:       训练出的回归树列表，长度为 n_estimators。
        train_loss_:  每轮训练后的训练集 MSE 列表，可用于画"逐轮残差
                      下降"曲线。
    """

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.init_ = 0.0
        self.trees_ = []
        self.train_loss_ = []

    def fit(self, X, y):
        """串行训练 n_estimators 棵回归树，每棵拟合当前残差。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。
            y: 回归目标，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # 初始预测：最省事的常数——y 的均值
        self.init_ = float(np.mean(y))
        current_pred = np.full(y.shape, self.init_, dtype=float)

        self.trees_ = []
        self.train_loss_ = []
        for _ in range(self.n_estimators):
            # 残差 = 真值 - 当前预测，代表"还差多少"
            residual = y - current_pred
            # 训练一棵回归树去拟合这份残差
            tree = _RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residual)
            # 按学习率把这棵树的纠错量累加进当前预测
            current_pred = current_pred + self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            # 记录这一轮的训练 MSE
            self.train_loss_.append(float(np.mean((y - current_pred) ** 2)))

        return self

    def predict(self, X):
        """预测 = 初始均值 + 学习率 × 各棵树预测之和。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            预测值数组，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X)
        return pred
