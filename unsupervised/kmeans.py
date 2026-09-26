"""K-Means 聚类——从零手写质心迭代（Lloyd 算法）。

K-Means 是最经典的无监督聚类算法。它不看标签，只凭点在空间里的
位置，把数据分成 k 个簇。核心是一个反复交替的两步循环：

1. 分配（assign）：把每个点划给离它最近的质心。
2. 更新（update）：把每个质心移到自己簇内所有点的均值位置。

这两步交替执行，直到质心不再移动（收敛）或达到最大轮数。可以证明
每一轮迭代都不会让"簇内平方和"（inertia）变大，所以算法必然收敛
到某个局部最优。整篇只用 NumPy。
"""
import numpy as np


class KMeans:
    """K-Means 聚类器，纯 NumPy 实现。

    参数：
        n_clusters: 簇的个数 k。
        max_iters:  最大迭代轮数，防止不收敛时死循环。
        n_init:     用不同随机初始化跑几次，保留 inertia 最小的那次。
                    K-Means 对初始质心敏感，单次初始化可能落进较差的
                    局部最优；多跑几次取最好的是标准做法（sklearn 同理）。
        seed:       随机种子，控制质心的初始化。

    训练后属性：
        centroids_: 最终质心，形状 (n_clusters, n_features)。
        labels_:    训练点的簇标签，形状 (n_samples,)。
        inertia_:   簇内平方和（每点到其质心的距离平方之和），
                    是肘部法则选 k 的依据。
        history_:   列表，记录最优那次运行每一轮（含初始）的质心位置，
                    供可视化迭代过程。
    """

    def __init__(self, n_clusters=3, max_iters=100, n_init=10, seed=0):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.n_init = n_init
        self.seed = seed
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.history_ = None

    def _assign(self, X, centroids):
        """把每个点分配到最近的质心，返回簇标签。

        用广播一次算出 (n_samples, n_clusters) 的距离矩阵：
        X[:, None, :] 形状 (n, 1, d)，centroids[None] 形状 (1, k, d)，
        相减后按最后一维求平方和，就是每个点到每个质心的平方距离。
        比较距离不用开平方，平方距离和欧氏距离单调一致。argmin 沿
        质心维取最近的那个。
        """
        diff = X[:, None, :] - centroids[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        return np.argmin(sq_dist, axis=1)

    def _inertia(self, X, centroids, labels):
        """计算簇内平方和：每个点到其所属质心的距离平方之和。"""
        diff = X - centroids[labels]
        return float(np.sum(diff * diff))

    def _fit_once(self, X, rng):
        """单次运行 Lloyd 算法，返回 (质心, 标签, inertia, 历史)。

        初始化时从训练点里随机不放回地抽 k 个作为初始质心（比在特征
        范围内随机撒点更稳，天然落在数据分布内）。每轮先分配再更新。
        更新质心时，若某个簇一个点都没分到（空簇），就保留它上一轮的
        位置，避免出现 NaN。当新旧质心完全相同（allclose）时提前收敛。
        """
        init_idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        centroids = X[init_idx].copy()

        history = [centroids.copy()]
        labels = self._assign(X, centroids)

        for _ in range(self.max_iters):
            # 更新：每个质心移到本簇均值；空簇保持原位
            new_centroids = centroids.copy()
            for c in range(self.n_clusters):
                mask = labels == c
                if np.any(mask):
                    new_centroids[c] = X[mask].mean(axis=0)

            history.append(new_centroids.copy())

            converged = np.allclose(new_centroids, centroids)
            centroids = new_centroids
            labels = self._assign(X, centroids)
            if converged:
                break

        inertia = self._inertia(X, centroids, labels)
        return centroids, labels, inertia, history

    def fit(self, X):
        """训练：多次随机初始化跑 Lloyd 算法，保留 inertia 最小的一次。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。
               K-Means 是无监督算法，只接收 X，不接收标签 y。

        返回：
            self。

        实现细节：
        K-Means 结果依赖初始质心，单次初始化可能收敛到较差的局部最优。
        这里用 n_init 个不同的随机初始化各跑一遍，比较各自的簇内平方和
        （inertia），把 inertia 最小的那次作为最终结果——这也是 sklearn
        默认的做法。history_ 保存的是被选中那次运行的完整质心轨迹。
        """
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)

        best = None
        for _ in range(self.n_init):
            centroids, labels, inertia, history = self._fit_once(X, rng)
            if best is None or inertia < best[2]:
                best = (centroids, labels, inertia, history)

        self.centroids_, self.labels_, self.inertia_, self.history_ = best
        return self

    def predict(self, X):
        """把新数据分配到已训练好的最近质心，返回簇标签。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            簇标签数组，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.centroids_)

    def fit_predict(self, X):
        """先 fit 再返回训练点的簇标签，等价于 fit(X).labels_。"""
        self.fit(X)
        return self.labels_
