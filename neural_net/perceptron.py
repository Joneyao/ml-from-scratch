"""感知机（Perceptron）——最小的神经元，从零手写。

感知机是神经网络的起点，可以看成"最小的神经元"：它把每个输入乘上
一个权重再求和，加上偏置得到一个得分，然后用一个阶跃函数把得分变成
0 或 1 的输出。整个结构就是"加权求和 + 阶跃激活"，和生物神经元
"输入信号累加、超过阈值就放电"的直觉一一对应。

它的学习规则也极其朴素：拿一个样本预测，猜错了就把权重往正确方向推
一点，猜对了就不动。这套规则叫"感知机学习规则"，是所有梯度式学习的
雏形。整篇只用 NumPy，不依赖 sklearn。

感知机有个著名的死穴：它只能画一条直线来分类，凡是线性不可分的问题
它都学不会。最经典的反例就是异或（XOR）——四个点里对角线同类、相邻
异类，一条直线无论怎么摆都分不开。这个局限直接导致了神经网络研究的
第一次寒冬，后来靠多层网络 + 反向传播才被突破。本模块提供 xor_data()
返回 XOR 的四个点，供测试和文章演示这个死穴。
"""
import numpy as np


class Perceptron:
    """单层感知机：加权求和 + 阶跃激活 + 感知机学习规则。

    属性：
        lr:      学习率，控制每次更新权重的步长。
        n_iters: 遍历整个训练集的轮数（epoch）。
        w:       权重向量，形状 (n_features,)，fit 后才有值。
        b:       偏置标量，fit 后才有值。
    """

    def __init__(self, lr=0.1, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _step(self, z):
        """阶跃激活函数：得分 > 0 输出 1，否则输出 0。

        这就是感知机的"放电"判定——加权和越过 0 这条阈值线就激活。
        """
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        """用感知机学习规则逐样本更新权重。

        参数：
            X: 特征矩阵，形状 (n_samples, n_features)。
            y: 标签，取值 {0, 1}，形状 (n_samples,)。

        学习规则：对每个样本先算预测 y_pred = step(w·x + b)，
        再按误差更新
            w += lr * (y_true - y_pred) * x
            b += lr * (y_true - y_pred)
        预测对了 (y_true - y_pred) 为 0，权重不动；预测错了误差为
        ±1，权重就朝减小错误的方向挪一步。整个训练把这套规则在数据
        上反复扫 n_iters 轮。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            for xi, target in zip(X, y):
                z = np.dot(xi, self.w) + self.b
                y_pred = 1 if z > 0 else 0
                update = self.lr * (target - y_pred)
                self.w += update * xi
                self.b += update
        return self

    def predict(self, X):
        """对输入做加权求和后过阶跃激活，返回预测标签 0/1。"""
        X = np.asarray(X, dtype=float)
        z = X @ self.w + self.b
        return self._step(z)


def xor_data():
    """返回异或（XOR）的四个点和标签，用于演示感知机的死穴。

    XOR 的真值表：
        (0, 0) -> 0
        (0, 1) -> 1
        (1, 0) -> 1
        (1, 1) -> 0

    这四个点在平面上呈"对角线同类、相邻异类"的分布，任何一条直线都
    无法把两类分开。单层感知机只会画直线，所以在这份数据上准确率上不了
    100%，一般最多分对 3/4。要学会 XOR，必须引入多层网络（隐藏层）。

    返回：
        X: 形状 (4, 2) 的坐标。
        y: 形状 (4,) 的标签 {0, 1}。
    """
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y
