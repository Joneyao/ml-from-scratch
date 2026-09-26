"""支持向量机（SVM）——从零手写 hinge loss + L2 正则 + 次梯度下降。

SVM 想找一条"最宽的分隔带"把两类分开。它不满足于随便一条能分开数据的
直线，而是要让离决策边界最近的那些点（支持向量）离边界尽量远。这个"最
大间隔"的目标写成优化问题，就是最小化 hinge loss 加 L2 正则：

    min  (1/2)·‖w‖²  +  C·Σ max(0, 1 - yᵢ(w·xᵢ + b))

前一项是 L2 正则，等价于最大化间隔（间隔宽度正比于 1/‖w‖）；后一项是
hinge loss，只惩罚"落在间隔带内或被分错"的点。C 越大越看重分对训练点，
C 越小越看重间隔宽、正则强。

hinge loss 在 yᵢ(w·xᵢ+b)=1 处有折点不可导，所以用次梯度下降：可导处用
梯度，折点处取一个次梯度即可。整篇只用 NumPy，不依赖 sklearn。
"""
import numpy as np


def rbf_kernel(X1, X2, gamma=1.0):
    """高斯 RBF 核矩阵：K[i,j] = exp(-gamma · ‖x1ᵢ - x2ⱼ‖²)。

    核技巧的核心思想：不显式把数据映射到高维空间，而是直接算映射后
    两点的内积。RBF 核衡量两点的相似度——距离越近越接近 1，越远越接近 0。
    这让线性不可分的数据（比如两个月牙形）在隐式的高维空间里变得可分。

    这里用平方距离的展开式做向量化计算，避免写双重循环：
        ‖a - b‖² = ‖a‖² + ‖b‖² - 2·a·b

    参数：
        X1:    形状 (m, d) 的样本矩阵。
        X2:    形状 (n, d) 的样本矩阵。
        gamma: 核宽度参数，越大核越"尖"（只有很近的点才相似）。

    返回：
        形状 (m, n) 的核矩阵。当 X1 与 X2 是同一批点时，对角线元素为
        exp(0) = 1（点到自身距离为 0）。
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    # 每个样本的平方范数，分别按行/列广播
    sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)   # (m, 1)
    sq2 = np.sum(X2**2, axis=1).reshape(1, -1)   # (1, n)
    # 平方距离矩阵，clip 掉浮点误差导致的微小负值
    sq_dist = sq1 + sq2 - 2.0 * (X1 @ X2.T)
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-gamma * sq_dist)


class LinearSVM:
    """线性支持向量机，纯 NumPy 次梯度下降实现。

    属性：
        w: 权重向量，形状 (n_features,)。
        b: 偏置标量。
    """

    def __init__(self):
        self.w = None
        self.b = 0.0
        # 内部特征标准化参数（fit 时计算），让不同量纲的特征收敛更快更稳
        self._mean = None
        self._std = None

    def fit(self, X, y, lr=0.01, n_iters=1000, C=1.0):
        """用次梯度下降最小化 hinge loss + L2 正则。

        参数：
            X:       特征矩阵，形状 (n_samples, n_features)。
            y:       标签，取值 {0, 1}，形状 (n_samples,)。
            lr:      学习率。
            n_iters: 迭代轮数。
            C:       惩罚系数，越大越不容忍分类错误。

        实现细节：
        1. 标签内部从 0/1 转成 ±1，因为 hinge loss 的推导基于 ±1 标签。
        2. 先把特征标准化（减均值除标准差），各维度尺度一致，梯度下降在
           固定学习率下更稳。训练完把参数折算回原始特征空间存进 w/b。

        次梯度推导：目标 J = (1/2)‖w‖² + C·Σ max(0, 1 - yᵢ·(w·xᵢ+b))。
        对某个样本，若 yᵢ·(w·xᵢ+b) >= 1（在间隔外且分对），hinge 项为 0，
        只有正则项贡献梯度：∂J/∂w = w，∂J/∂b = 0；否则该样本还额外贡献
        -C·yᵢ·xᵢ（对 w）和 -C·yᵢ（对 b）。这里用批量平均形式更新。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # 标签 0/1 -> ±1
        y_pm = np.where(y <= 0, -1.0, 1.0)

        # 标准化：std 为 0 的常数列用 1 兜底避免除零
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std = np.where(self._std == 0.0, 1.0, self._std)
        Xs = (X - self._mean) / self._std

        # 在标准化空间里做次梯度下降
        w = np.zeros(n_features)
        b = 0.0
        for _ in range(n_iters):
            margins = y_pm * (Xs @ w + b)   # yᵢ·(w·xᵢ+b)
            mask = margins < 1.0            # 违反间隔约束的样本
            # 权重梯度：正则项 w，加上违反约束样本的 hinge 贡献（批量平均）
            grad_w = w - C * (Xs[mask].T @ y_pm[mask]) / n_samples
            grad_b = -C * np.sum(y_pm[mask]) / n_samples
            w -= lr * grad_w
            b -= lr * grad_b

        # 把标准化空间参数折算回原始特征空间：
        # score = w·(x-mean)/std + b = (w/std)·x + (b - w·mean/std)
        self.w = w / self._std
        self.b = b - np.sum(w * self._mean / self._std)
        return self

    def decision_function(self, X):
        """返回原始得分 w·x + b，符号决定类别，绝对值反映离边界的远近。"""
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X):
        """返回预测标签 0/1：得分 >= 0 判为正类（原始标签 1）。"""
        return (self.decision_function(X) >= 0.0).astype(int)
