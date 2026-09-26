"""异常检测——从零手写两种经典方法（高斯密度 + Z-score）。

异常检测要回答一个朴素的问题：这个点是不是"太少见了"？直觉上，正常
数据会扎堆出现在某个区域，而异常点孤零零地落在人烟稀少的地方。把这个
直觉翻译成数学，就是一句话：

    异常 = 少见 = 概率低。

我们先用正常数据估计出"数据长什么样"（比如它服从什么分布），再对每个
新样本算它的出现概率。概率高说明这个点和大家一样普通；概率低说明它是
个稀客，很可能是异常。两种方法都是这个思路的具体实现：

- GaussianAnomalyDetector：假设每个特征服从高斯分布，直接算样本的概率
  密度，密度低于阈值就判异常。
- ZScoreDetector：更简单，只看样本偏离均值多少个标准差，偏太远就异常。

约定输出对齐 sklearn 的 IsolationForest / OneClassSVM：predict 返回
1 表示正常、-1 表示异常。整篇只用 NumPy。
"""
import numpy as np


class GaussianAnomalyDetector:
    """基于高斯分布的异常检测器，纯 NumPy 实现。

    核心假设：正常数据的每个特征各自服从一维高斯分布，且特征之间相互
    独立。于是一个样本的联合概率密度就是各维高斯密度的连乘。异常点落在
    分布的尾巴上，联合密度会非常小——把密度低于某个阈值的点判为异常。

    为什么用对数密度而不是密度本身？高维时每一维的密度都小于 1，几十维
    连乘会下溢成 0，浮点数存不住。取对数把连乘变成求和，既避免下溢，又
    保持单调性（密度越小对数越小），判别结果完全等价。

    阈值怎么定？我们不凭空拍一个数，而是用 contamination（预期异常占比）
    反推：在训练数据上算出所有样本的对数密度，取第 contamination 分位数
    作为阈值。比如 contamination=0.05，就把训练集里对数密度最低的那 5%
    当成异常的临界线。这样阈值是数据驱动的，换个数据集也能自适应。

    参数：
        contamination: 预期异常样本占比，用来定阈值，默认 0.05。
        eps:           方差的下限，防止某一维方差为 0 时除零，默认 1e-9。

    训练后属性：
        mu_:        各特征均值，形状 (n_features,)。
        var_:       各特征方差，形状 (n_features,)。
        threshold_: 对数密度阈值，样本对数密度低于它判为异常。
    """

    def __init__(self, contamination=0.05, eps=1e-9):
        self.contamination = contamination
        self.eps = eps
        self.mu_ = None
        self.var_ = None
        self.threshold_ = None

    def _log_density(self, X):
        """算每个样本的联合对数概率密度（假设各维高斯且独立）。

        一维高斯的概率密度是
            p(x) = 1/sqrt(2*pi*var) * exp(-(x-mu)^2 / (2*var))
        取对数得
            log p(x) = -0.5 * [ log(2*pi*var) + (x-mu)^2 / var ]
        各维独立时联合密度是连乘，取对数就变成各维对数密度求和。所以对
        每个样本，把它在各维上的 log p 沿特征轴加起来，就是联合对数密度。
        """
        X = np.asarray(X, dtype=float)
        # (x - mu)^2 / var，逐元素；再拼上归一化常数项
        z2 = (X - self.mu_) ** 2 / self.var_
        log_p_per_dim = -0.5 * (np.log(2.0 * np.pi * self.var_) + z2)
        # 沿特征轴求和 = 各维密度连乘取对数 = 联合对数密度
        return np.sum(log_p_per_dim, axis=1)

    def fit(self, X):
        """训练：估计各维均值/方差，并按 contamination 反推对数密度阈值。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。异常检测通常用
               "大部分正常"的数据训练，算法本身不需要标签。

        返回：
            self。

        实现细节：
        均值、方差按列（每个特征）估计。方差用总体方差（除以 N），并加上
        eps 下限防止某维恒定时方差为 0 导致后续除零。阈值取训练集对数密度
        的第 contamination 分位数——即预期有这么大比例的点会低于它被判异常。
        """
        X = np.asarray(X, dtype=float)
        self.mu_ = X.mean(axis=0)
        self.var_ = X.var(axis=0) + self.eps

        log_dens = self._log_density(X)
        # contamination 分位数：低于此值的样本约占 contamination 比例
        self.threshold_ = float(np.quantile(log_dens, self.contamination))
        return self

    def score_samples(self, X):
        """返回每个样本的对数概率密度，值越大越"正常"。

        参数：
            X: 待评分特征矩阵，形状 (n_samples, n_features)。

        返回：
            对数密度数组，形状 (n_samples,)。这个方向和 sklearn 的
            score_samples 一致：分数越高越正常，越低越异常。
        """
        return self._log_density(X)

    def predict(self, X):
        """判定每个样本正常还是异常：对数密度低于阈值即为异常。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            数组，1 表示正常、-1 表示异常（对齐 sklearn 约定）。
        """
        log_dens = self._log_density(X)
        # 密度 >= 阈值判正常(1)，否则异常(-1)
        return np.where(log_dens >= self.threshold_, 1, -1)


class ZScoreDetector:
    """基于 Z-score（标准分数）的异常检测器，纯 NumPy 实现。

    这是最直观的异常检测：一个值离均值有多远，用"几个标准差"来量。这个
    倍数就是 Z-score：
        z = (x - mu) / sigma
    经验上，正态数据约 99.7% 落在均值 ±3 个标准差内（三西格玛法则）。所以
    只要某个样本在任意一个特征上的 |z| 超过 k（默认 3），就认为它在那一维
    上偏离得太离谱，判为异常。

    和高斯密度法的区别：高斯法把各维证据连乘成一个联合概率再统一判断；
    Z-score 法则是"逐维独立看，任一维出格就算异常"，更保守也更简单，不
    假设特征独立地相乘，只做单维阈值判断。多维时采用 any 逻辑：只要有一个
    特征越界，整条样本就是异常。

    参数：
        k: 判异常的标准差倍数阈值，默认 3.0。

    训练后属性：
        mu_:    各特征均值，形状 (n_features,)。
        sigma_: 各特征标准差，形状 (n_features,)。
    """

    def __init__(self, k=3.0, eps=1e-9):
        self.k = k
        self.eps = eps
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X):
        """训练：估计各特征的均值和标准差。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。

        返回：
            self。

        标准差加 eps 下限，防止某维恒定（sigma=0）时算 z 除零。
        """
        X = np.asarray(X, dtype=float)
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + self.eps
        return self

    def predict(self, X):
        """判定正常/异常：任一特征的 |z| 超过 k 即为异常。

        参数：
            X: 待预测特征矩阵，形状 (n_samples, n_features)。

        返回：
            数组，1 表示正常、-1 表示异常（对齐 sklearn 约定）。
        """
        X = np.asarray(X, dtype=float)
        z = np.abs((X - self.mu_) / self.sigma_)
        # 逐维判断是否越界，再沿特征轴取 any：只要有一维出格就算异常
        is_anomaly = np.any(z > self.k, axis=1)
        return np.where(is_anomaly, -1, 1)
