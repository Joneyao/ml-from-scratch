"""K 近邻（KNN）——从零手写欧氏距离 + 多数投票。

KNN 是最直观的分类算法，甚至没有"训练"这一步：fit 阶段只是把训练
数据原样存下来（惰性学习）。真正的计算发生在 predict 阶段——对每个
待预测点，算它到所有训练点的欧氏距离，挑出最近的 k 个邻居，让这 k
个邻居的标签投票，票数最多的类别就是预测结果。整篇只用 NumPy。
"""
import numpy as np


class KNN:
    """K 近邻分类器，纯 NumPy 实现。

    属性：
        k:  参与投票的邻居个数。
        X_: 存下的训练特征，形状 (n_samples, n_features)。
        y_: 存下的训练标签，形状 (n_samples,)。
    """

    def __init__(self, k=5):
        self.k = k
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        """惰性学习：只把训练数据存下来，不做任何计算。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。
            y: 标签，形状 (n_samples,)。
        """
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        """对每个样本做预测：欧氏距离 → 取最近 k 个 → 多数投票。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            预测标签数组，形状 (n_samples,)。

        实现细节：
        对每个待预测点，先算它到全部训练点的欧氏距离。这里比较距离
        大小时不需要开平方——平方距离和欧氏距离单调一致，省掉 sqrt。
        用 argsort 把距离从小到大排序，取前 k 个下标就是最近的 k 个
        邻居。再用 np.bincount 统计这 k 个邻居各标签出现的次数，argmax
        取票数最多的类别。平票时 argmax 取最小的标签，行为确定。
        """
        X = np.asarray(X, dtype=float)
        preds = np.empty(X.shape[0], dtype=self.y_.dtype)
        for i, x in enumerate(X):
            # 到所有训练点的平方欧氏距离，形状 (n_train,)
            diff = self.X_ - x
            sq_dist = np.sum(diff * diff, axis=1)
            # 最近 k 个邻居的下标
            knn_idx = np.argsort(sq_dist)[: self.k]
            knn_labels = self.y_[knn_idx]
            # 多数投票：统计每个标签的票数，取最多的
            preds[i] = np.bincount(knn_labels).argmax()
        return preds
