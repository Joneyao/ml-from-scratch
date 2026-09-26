"""逻辑回归——从零手写 sigmoid + 交叉熵损失 + 梯度下降。

逻辑回归是最基础的二分类模型。它先算一个线性得分 z = X·w + b，
再用 sigmoid 把得分压到 (0, 1) 当作正类概率，最后用梯度下降最小化
交叉熵损失来学参数。整篇只用 NumPy，不依赖 sklearn。
"""
import numpy as np


def sigmoid(z):
    """数值稳定的 sigmoid：1 / (1 + exp(-z))。

    直接算 exp(-z) 在 z 很大的负数时会溢出。这里分正负两支：
    - z >= 0：用 1 / (1 + exp(-z))，此时 exp(-z) 不会溢出。
    - z < 0：用 exp(z) / (1 + exp(z))，此时 exp(z) 不会溢出。
    两支数学上等价，但各自避开了 exp 的溢出区。
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    # z >= 0 分支
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    # z < 0 分支
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out[()] if out.ndim == 0 else out


def cross_entropy_loss(y_true, y_prob, eps=1e-12):
    """二分类交叉熵损失（平均）。

    L = -mean( y·log(p) + (1-y)·log(1-p) )。用 eps 把概率夹在
    (eps, 1-eps) 里，避免 log(0) 出现 -inf。
    """
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


class LogisticRegression:
    """二分类逻辑回归，纯 NumPy 梯度下降实现。

    属性：
        coef_:      权重向量，形状 (n_features,)。
        intercept_: 偏置标量。
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0
        # 内部特征标准化参数（fit 时计算），让不同量纲的特征收敛更快更稳
        self._mean = None
        self._std = None

    def fit(self, X, y, lr=0.1, n_iters=1000):
        """用批量梯度下降拟合参数。

        参数：
            X:       特征矩阵，形状 (n_samples, n_features)。
            y:       标签，取值 {0, 1}，形状 (n_samples,)。
            lr:      学习率。
            n_iters: 迭代轮数。

        实现细节：先把特征标准化（减均值除标准差），这样各维度尺度
        一致，梯度下降在固定学习率下收敛更快也更稳。训练完成后把标准
        化的缩放折算回原始特征空间，存进 coef_/intercept_，因此外部
        用原始 X 调 predict 即可，无需再手动标准化。

        梯度推导：对交叉熵损失求导后，权重梯度是
        (1/n)·Xᵀ·(p - y)，偏置梯度是 (p - y) 的均值。
        每轮沿负梯度方向更新一步。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        # 标准化：记住均值和标准差，std 为 0 的常数列用 1 兜底避免除零
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std = np.where(self._std == 0.0, 1.0, self._std)
        Xs = (X - self._mean) / self._std

        # 在标准化空间里做梯度下降
        w = np.zeros(n_features)
        b = 0.0
        for _ in range(n_iters):
            z = Xs @ w + b
            p = sigmoid(z)
            error = p - y  # 预测概率与真实标签的差
            grad_w = (Xs.T @ error) / n_samples
            grad_b = np.mean(error)
            w -= lr * grad_w
            b -= lr * grad_b

        # 把标准化空间的参数折算回原始特征空间：
        # z = w·(x-mean)/std + b = (w/std)·x + (b - w·mean/std)
        self.coef_ = w / self._std
        self.intercept_ = b - np.sum(w * self._mean / self._std)
        return self

    def predict_proba(self, X):
        """返回样本属于正类（标签 1）的概率，形状 (n_samples,)。"""
        X = np.asarray(X, dtype=float)
        z = X @ self.coef_ + self.intercept_
        return sigmoid(z)

    def predict(self, X):
        """返回预测标签 0/1，概率 >= 0.5 判为正类。"""
        return (self.predict_proba(X) >= 0.5).astype(int)
