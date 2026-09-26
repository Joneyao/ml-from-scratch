"""多层感知机（MLP）——纯 NumPy 从零实现，识别 8x8 手写数字。

## MLP 是什么

单个感知机只能画一条直线，处理线性可分的问题。
把很多感知机**堆叠**成层，层与层之间全连接，就得到多层感知机：

    输入层(64) → 隐藏层(hidden_size, ReLU) → 输出层(10, softmax)

隐藏层里的每个神经元都是一个感知机（加权求和 + 激活）。
多个感知机并排组成一层，多层再串起来，中间夹上非线性激活（ReLU），
网络就能拟合任意复杂的非线性决策边界——这就是"多层"带来的表达力。

## 怎么训练：前向 + 反向传播

- **前向传播**：数据从输入流到输出，逐层做"矩阵乘 + 加偏置 + 激活"，
  最后 softmax 把 10 个输出变成一个概率分布，用交叉熵衡量它和真实标签的差距。
- **反向传播**：把这个差距（损失）对每个权重求偏导。利用链式法则，
  梯度从输出层沿着网络**反向**逐层流回，算出每个 W、b 该往哪个方向调。
- **梯度下降**：所有参数各自减去"学习率 × 梯度"，损失下降一点点。
  反复迭代很多个 epoch，网络就学会了识别数字。

softmax + 交叉熵有个漂亮的性质：输出层的梯度就是 (预测概率 - 真实 one-hot)，
形式极简，这也是分类任务几乎总用这对组合的原因。
"""
import numpy as np


class MLP:
    """单隐藏层的多层感知机分类器（纯 NumPy 手写前向 + 反向）。

    结构：输入(64) → 隐藏层(hidden_size, ReLU) → 输出(10, softmax)。

    参数
    ----
    hidden_size : int
        隐藏层神经元个数（= 并排堆叠的感知机个数）。
    lr : float
        学习率，梯度下降每步的步长。
    n_epochs : int
        训练轮数，整个训练集被完整遍历的次数（这里用全批量梯度下降）。
    seed : int
        随机种子，控制权重初始化，保证结果可复现。

    训练后可读属性
    --------------
    loss_history_ : list[float]
        每个 epoch 的训练交叉熵损失（用于画训练曲线）。
    acc_history_ : list[float]
        每个 epoch 的训练集准确率。
    """

    def __init__(self, hidden_size=64, lr=0.1, n_epochs=100, seed=0):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.loss_history_ = []
        self.acc_history_ = []

    # ---- 内部工具函数 ----
    @staticmethod
    def _normalize(X):
        """归一化：digits 像素值在 0-16，除以 16 缩放到 [0, 1]。

        输入若不是 [0,1] 量级，ReLU 前的加权和会过大，训练不稳定。
        """
        return np.asarray(X, dtype=np.float64) / 16.0

    @staticmethod
    def _one_hot(y, n_classes=10):
        """把整数标签 (n,) 转成 one-hot 矩阵 (n, n_classes)。"""
        y = np.asarray(y, dtype=np.int64)
        onehot = np.zeros((y.shape[0], n_classes), dtype=np.float64)
        onehot[np.arange(y.shape[0]), y] = 1.0
        return onehot

    @staticmethod
    def _relu(z):
        """ReLU 激活：max(0, z)，给网络引入非线性。"""
        return np.maximum(0.0, z)

    @staticmethod
    def _softmax(z):
        """按行做 softmax，把每行打分变成一个概率分布。

        减去每行最大值是数值稳定技巧，防止 exp 溢出（结果不变）。
        """
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _forward(self, Xn):
        """前向传播，返回中间量供反向传播复用。

        z1 = Xn·W1 + b1     ->  a1 = ReLU(z1)   （隐藏层）
        z2 = a1·W2 + b2     ->  probs = softmax(z2)  （输出层）
        """
        z1 = Xn @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self._softmax(z2)
        return z1, a1, z2, probs

    def fit(self, X, y):
        """训练：全批量的前向 + 交叉熵损失 + 反向传播 + 梯度下降。

        y 是 0-9 的整数标签，内部转成 one-hot 参与损失计算。
        每个 epoch 记录一次训练损失和准确率到 loss_history_ / acc_history_。
        """
        rng = np.random.default_rng(self.seed)
        Xn = self._normalize(X)
        Y = self._one_hot(y, n_classes=10)
        y = np.asarray(y, dtype=np.int64)

        n_samples, n_features = Xn.shape
        n_classes = 10
        h = self.hidden_size

        # He 初始化：适配 ReLU，权重方差按 2/fan_in 缩放，避免梯度消失/爆炸
        self.W1 = rng.standard_normal((n_features, h)) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(h)
        self.W2 = rng.standard_normal((h, n_classes)) * np.sqrt(2.0 / h)
        self.b2 = np.zeros(n_classes)

        self.loss_history_ = []
        self.acc_history_ = []

        for _ in range(self.n_epochs):
            # ---- 前向 ----
            z1, a1, z2, probs = self._forward(Xn)

            # ---- 交叉熵损失（加 1e-12 防 log(0)）----
            loss = -np.sum(Y * np.log(probs + 1e-12)) / n_samples

            # ---- 反向传播（手写梯度）----
            # softmax + 交叉熵合并求导，输出层误差 = 预测概率 - 真实 one-hot
            dz2 = (probs - Y) / n_samples          # (n, 10)
            dW2 = a1.T @ dz2                        # (h, 10)
            db2 = np.sum(dz2, axis=0)              # (10,)

            da1 = dz2 @ self.W2.T                   # (n, h)
            dz1 = da1 * (z1 > 0)                    # ReLU 导数：z1>0 处为 1
            dW1 = Xn.T @ dz1                         # (features, h)
            db1 = np.sum(dz1, axis=0)              # (h,)

            # ---- 梯度下降更新 ----
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            # ---- 记录训练曲线 ----
            train_pred = np.argmax(probs, axis=1)
            acc = float(np.mean(train_pred == y))
            self.loss_history_.append(float(loss))
            self.acc_history_.append(acc)

        return self

    def predict(self, X):
        """预测：前向传播取 softmax 概率最大的类别，返回 0-9 整数标签。"""
        Xn = self._normalize(X)
        _, _, _, probs = self._forward(Xn)
        return np.argmax(probs, axis=1).astype(np.int64)
