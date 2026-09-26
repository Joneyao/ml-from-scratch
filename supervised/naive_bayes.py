"""朴素贝叶斯——从零手写高斯与多项式两种变体。

朴素贝叶斯基于贝叶斯定理，外加一个"朴素"假设：给定类别时，各特征
相互条件独立。于是类后验正比于先验乘以各特征的条件概率连乘。分类时
比较各类的后验大小，取最大者。为避免大量小概率连乘下溢，全程在对数
空间累加：log P(y|x) ∝ log P(y) + Σ log P(x_i|y)。整篇只用 NumPy。

两个变体的区别在于对 P(x_i|y) 的建模：
- GaussianNB：特征连续，假设每类每维服从高斯分布，用概率密度。
- MultinomialNB：特征是计数（如词频），用带拉普拉斯平滑的多项式模型。
"""
import numpy as np


class GaussianNB:
    """高斯朴素贝叶斯，适用于连续特征。

    对每个类别的每个特征，估计训练样本的均值和方差，假设该特征在
    该类下服从高斯分布。预测时用高斯概率密度的对数作为条件对数概率，
    与类先验的对数相加，取后验最大的类。

    属性：
        classes_:  类别标签数组，形状 (n_classes,)。
        theta_:    各类各特征的均值，形状 (n_classes, n_features)。
        var_:      各类各特征的方差，形状 (n_classes, n_features)。
        log_prior_: 各类先验的对数，形状 (n_classes,)。
    """

    def __init__(self, var_smoothing=1e-9):
        # var_smoothing：给方差加一个极小值，避免某维方差为 0 时除零/取 log 出问题
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.theta_ = None
        self.var_ = None
        self.log_prior_ = None

    def fit(self, X, y):
        """按类分组，估计每类每特征的均值、方差和类先验。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。
            y: 类别标签，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.log_prior_ = np.zeros(n_classes)

        # 用全局最大方差乘以平滑系数作为方差下限，和 sklearn 的做法一致
        epsilon = self.var_smoothing * X.var(axis=0).max()

        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[idx] = Xc.mean(axis=0)
            self.var_[idx] = Xc.var(axis=0) + epsilon
            # 先验用该类样本占比，取对数
            self.log_prior_[idx] = np.log(Xc.shape[0] / X.shape[0])
        return self

    def _joint_log_likelihood(self, X):
        """算每个样本对每个类的联合对数概率 log P(y) + Σ log P(x_i|y)。

        高斯对数密度：log N(x; μ, σ²) = -0.5·log(2πσ²) - (x-μ)²/(2σ²)。
        对每个特征求和，再加上类先验的对数。返回形状 (n_samples, n_classes)。
        """
        X = np.asarray(X, dtype=float)
        jll = np.zeros((X.shape[0], len(self.classes_)))
        for idx in range(len(self.classes_)):
            mean = self.theta_[idx]
            var = self.var_[idx]
            # 常数项：每维 -0.5*log(2πσ²)，对特征求和
            log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            # 指数项：每维 -(x-μ)²/(2σ²)，对特征求和
            log_exp = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            jll[:, idx] = self.log_prior_[idx] + log_norm + log_exp
        return jll

    def predict(self, X):
        """返回每个样本后验最大的类别标签，形状 (n_samples,)。"""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]


class MultinomialNB:
    """多项式朴素贝叶斯，适用于离散计数特征（如词频），常用于文本分类。

    把每个类别看成一个多项式分布，特征值是各"词"出现的次数。用训练集里
    该类下每个词的总次数估计词概率，并加拉普拉斯平滑避免零概率（某词在
    某类训练样本中从未出现时，概率不至于变成 0 而抹掉整条连乘）。

    属性：
        classes_:        类别标签数组，形状 (n_classes,)。
        feature_log_prob_: 各类各特征的对数条件概率，形状 (n_classes, n_features)。
        class_log_prior_:  各类先验的对数，形状 (n_classes,)。
    """

    def __init__(self, alpha=1.0):
        # alpha：拉普拉斯（加性）平滑系数，alpha=1.0 即经典的"加一平滑"
        self.alpha = alpha
        self.classes_ = None
        self.feature_log_prob_ = None
        self.class_log_prior_ = None

    def fit(self, X, y):
        """统计每类的词总数，加平滑后估计对数条件概率与类先验。

        参数：
            X: 计数特征矩阵（非负整数/浮点），形状 (n_samples, n_features)。
            y: 类别标签，形状 (n_samples,)。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_log_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            # 该类下每个词的总出现次数
            count = Xc.sum(axis=0)
            # 拉普拉斯平滑：分子加 alpha，分母加 alpha*词表大小
            smoothed = count + self.alpha
            total = smoothed.sum()
            self.feature_log_prob_[idx] = np.log(smoothed / total)
            # 先验用该类样本占比
            self.class_log_prior_[idx] = np.log(Xc.shape[0] / X.shape[0])
        return self

    def _joint_log_likelihood(self, X):
        """算联合对数概率 log P(y) + Σ x_i·log P(word_i|y)。

        多项式模型里，某词出现 x_i 次贡献 x_i·log P(word_i|y)，所以整体
        是计数矩阵与对数条件概率的点积，再加类先验。返回 (n_samples, n_classes)。
        """
        X = np.asarray(X, dtype=float)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        """返回每个样本后验最大的类别标签，形状 (n_samples,)。"""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
