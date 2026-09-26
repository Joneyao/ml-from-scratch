"""主成分分析（PCA）——从零手写协方差 + 特征值分解。

PCA 干的事就一句话：找数据里方差最大的那几个方向。

想象一团散在高维空间里的点。这团点在某些方向上铺得很开（方差大），
在另一些方向上挤成一条缝（方差小）。方差大的方向携带了数据的主要
差异信息，方差小的方向近似是噪声或冗余。PCA 把这些"最开阔"的方向
挑出来当新坐标轴，再把数据投影上去，就用更少的维度保住了大部分信息。

怎么找这些方向？中心化后的数据，其协方差矩阵刻画了各维度间的散布。
对协方差矩阵做特征值分解：特征向量就是一个个正交的候选方向，对应的
特征值恰好等于数据沿这个方向的方差。于是"方差最大的方向"就是最大
特征值对应的特征向量。取前 n_components 大的，就是前几个主成分。

整篇只用 NumPy。
"""
import numpy as np


class PCA:
    """主成分分析降维器，纯 NumPy 实现。

    属性：
        n_components:               保留的主成分个数。
        mean_:                      训练数据每一维的均值，形状 (n_features,)，
                                    transform 时要先减掉它做中心化。
        components_:                主成分方向，形状 (n_components, n_features)，
                                    每一行是一个单位特征向量（新坐标轴）。
        explained_variance_:        每个主成分解释的方差（对应特征值），
                                    形状 (n_components,)。
        explained_variance_ratio_:  每个主成分解释的方差占全体方差的比例，
                                    形状 (n_components,)，降序排列。
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """学习主成分：中心化 → 协方差 → 特征值分解 → 取最大的几个方向。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。

        实现细节：
        1) 中心化——减去每一维的均值，让数据以原点为中心。协方差的定义
           本就建立在"偏离均值多少"之上，不减均值算出来的是围绕原点的
           二阶矩，不是方差。
        2) 协方差矩阵——np.cov 默认按行是变量、列是观测，这里数据是
           每行一个样本，所以传 rowvar=False，得到 (n_features, n_features)
           的对称矩阵。
        3) 特征值分解——协方差矩阵是实对称矩阵，用 np.linalg.eigh 而不是
           eig：eigh 专门针对对称矩阵，返回实特征值且按升序排列，数值更
           稳定。特征向量在列上，第 i 列对应第 i 个特征值。
        4) 排序取大——eigh 给的是升序，翻转成降序，特征值越大代表数据沿
           该方向铺得越开（方差越大）。取前 n_components 个特征向量当主
           成分，转成每行一个方向存进 components_。
        5) 方差比例——特征值就是各方向的方差，除以全部特征值之和即为
           每个主成分解释的方差占比。
        """
        X = np.asarray(X, dtype=float)

        # 1) 中心化：记住均值，减去均值
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # 2) 协方差矩阵，形状 (n_features, n_features)
        cov = np.cov(X_centered, rowvar=False)

        # 3) 实对称矩阵的特征值分解，eigvals 升序，eigvecs 按列
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 4) 翻转成降序，取前 n_components 个
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        top_vals = eigvals[: self.n_components]
        top_vecs = eigvecs[:, : self.n_components]

        # 每行一个主成分方向
        self.components_ = top_vecs.T
        self.explained_variance_ = top_vals

        # 5) 方差解释比例：各主成分方差 / 全体方差之和
        total_var = eigvals.sum()
        self.explained_variance_ratio_ = top_vals / total_var

        return self

    def transform(self, X):
        """把数据投影到已学到的主成分上，完成降维。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。

        返回：
            降维后的数据，形状 (n_samples, n_components)。

        投影的本质是求内积：中心化后的样本和每个主成分方向做点积，得到
        它在这个新坐标轴上的坐标。X_centered @ components_.T 一次算出所有
        样本在所有主成分上的坐标。
        """
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """先 fit 学主成分，再 transform 投影，一步到位。"""
        self.fit(X)
        return self.transform(X)
