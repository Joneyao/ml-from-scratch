"""随机森林（分类）——从零手写 Bagging + 特征随机 + 多数投票。

一棵决策树容易过拟合：它把训练集里的噪声也当成规律学了进去。随机
森林的思路是"三个臭皮匠"——训练很多棵有差异的树，让它们各自投票，
再取多数。只要每棵树的错误方式不完全一样，投票就能把彼此的随机错误
互相抵消，整体比单棵树更稳。

制造差异靠两个随机来源：

1. Bagging（自助采样）：每棵树不是拿全部训练数据，而是从训练集里
   有放回地随机抽同样多的样本。有放回意味着有的样本被抽中多次、有的
   一次没抽中，于是每棵树看到的数据都略有不同。

2. 特征随机：每棵树训练时只随机看一部分特征列（这里取 sqrt(n_features)
   个），进一步打散树之间的相关性——否则所有树都会盯着那几个最强特征，
   长得几乎一样，投票就失去意义。因为每棵树只用了部分列，预测时要把
   输入按同一批列切出来再喂给对应的树。

预测阶段，每棵树各出一票预测类别，森林统计所有票数，多数类胜出。
整篇只依赖 NumPy 和已有的 DecisionTree 基学习器。
"""
import numpy as np

from supervised.decision_tree import DecisionTree


class RandomForest:
    """随机森林分类器，纯 NumPy 实现（基学习器复用 DecisionTree）。

    参数：
        n_trees:      森林里树的数量，越多通常越稳但越慢。
        max_depth:    每棵基决策树的最大深度（预剪枝）。
        sample_ratio: 每棵树自助采样的样本数占训练集的比例，默认 1.0
                      表示抽和训练集一样多的样本（标准 Bagging）。
        seed:         随机种子，保证自助采样与特征选择可复现。

    属性：
        trees_:     训练好的基决策树列表。
        features_:  与 trees_ 一一对应，每棵树用到的特征列下标数组。
        classes_:   训练集中出现过的类别（升序）。
    """

    def __init__(self, n_trees=10, max_depth=5, sample_ratio=1.0, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.trees_ = None
        self.features_ = None
        self.classes_ = None

    def fit(self, X, y):
        """训练森林：为每棵树做自助采样 + 特征随机，再各自建树。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。
            y: 标签，形状 (n_samples,)，须为非负整数类别。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        # 每棵树随机选用的特征列数：sqrt(n_features)，至少 1 列
        n_feat_sub = max(1, int(np.sqrt(n_features)))
        # 每棵树自助采样的样本数
        n_sub = max(1, int(round(n_samples * self.sample_ratio)))

        self.trees_ = []
        self.features_ = []
        for _ in range(self.n_trees):
            # 1) Bagging：有放回抽样得到样本子集
            sample_idx = rng.integers(0, n_samples, size=n_sub)
            # 2) 特征随机：无放回选一部分特征列
            feat_idx = rng.choice(n_features, size=n_feat_sub, replace=False)
            feat_idx = np.sort(feat_idx)

            X_sub = X[sample_idx][:, feat_idx]
            y_sub = y[sample_idx]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sub, y_sub)

            self.trees_.append(tree)
            self.features_.append(feat_idx)

        return self

    def predict(self, X):
        """所有树对每个样本各投一票，返回得票最多的类别。

        每棵树只认得自己训练时用过的那批特征列，所以预测时先把 X 按
        对应列切出来再交给该树。收集全部树的预测后，逐样本统计每个类别
        的票数，取票数最多的类别作为最终预测（平票时取较小的类别标签，
        行为确定）。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            预测标签数组，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        # 收集每棵树的预测，形状 (n_trees, n_samples)
        all_preds = np.empty((self.n_trees, n_samples), dtype=self.classes_.dtype)
        for i, (tree, feat_idx) in enumerate(zip(self.trees_, self.features_)):
            all_preds[i] = tree.predict(X[:, feat_idx])

        # 逐样本做多数投票：统计每个候选类别的票数，取最多者
        n_classes = self.classes_.shape[0]
        votes = np.zeros((n_samples, n_classes), dtype=int)
        for ci, cls in enumerate(self.classes_):
            votes[:, ci] = np.sum(all_preds == cls, axis=0)
        winner_col = np.argmax(votes, axis=1)  # 平票取最小下标（类别升序）
        return self.classes_[winner_col]
