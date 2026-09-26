"""纯 NumPy 手写 MLP 识别手写数字的测试。

数据集用 sklearn 的 load_digits（8x8=64 维，10 分类，1797 张），
轻量、无需下载大 MNIST。sklearn 只负责加载数据和切分训练/测试集，
MLP 的前向、反向、softmax、梯度下降全部由 neural_net.mlp_mnist 手写实现。
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neural_net.mlp_mnist import MLP


def _load_split():
    """加载 digits 数据并按 8:2 切分为训练/测试集（固定随机种子，结果可复现）。"""
    digits = load_digits()
    X, y = digits.data, digits.target
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_fit_predict_shape_and_range():
    """MLP 能 fit/predict，predict 输出形状正确、类别落在 0-9 范围内。"""
    X_train, X_test, y_train, y_test = _load_split()
    # 用较小的训练轮数快速验证接口，不追求精度
    clf = MLP(hidden_size=32, lr=0.1, n_epochs=10, seed=0)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    # 形状：每个样本一个预测标签
    assert pred.shape == (X_test.shape[0],)
    # 取值范围：预测的数字类别必须在 0-9 之间
    assert pred.min() >= 0
    assert pred.max() <= 9
    # 类型：整数标签
    assert np.issubdtype(pred.dtype, np.integer)


def test_test_accuracy_above_90_percent():
    """在 load_digits 上训练后，测试集准确率 > 0.90。

    纯 NumPy 手写的单隐藏层 MLP 就能把手写数字识别到 90%+。
    """
    X_train, X_test, y_train, y_test = _load_split()
    clf = MLP(hidden_size=64, lr=0.2, n_epochs=200, seed=0)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = float(np.mean(pred == y_test))
    assert acc > 0.90, f"测试集准确率仅 {acc:.4f}，未达到 0.90"


def test_loss_history_decreases():
    """训练损失整体下降：末尾 epoch 的损失应显著小于开头的损失。"""
    X_train, X_test, y_train, y_test = _load_split()
    clf = MLP(hidden_size=64, lr=0.2, n_epochs=100, seed=0)
    clf.fit(X_train, y_train)

    # 记录了每个 epoch 的损失与准确率
    assert len(clf.loss_history_) == 100
    assert len(clf.acc_history_) == 100
    # 整体下降：最后的损失 < 开头的损失
    assert clf.loss_history_[-1] < clf.loss_history_[0]
    # 准确率随训练提升：末尾准确率 >= 开头准确率
    assert clf.acc_history_[-1] >= clf.acc_history_[0]
