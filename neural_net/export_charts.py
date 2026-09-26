"""运行神经网络模块，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt

from common.plotting import setup_cjk_font, save_fig, plot_decision_boundary
from common.datasets import make_blobs_2d
from neural_net.perceptron import Perceptron, xor_data
from neural_net.mlp_mnist import MLP

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_e1_perceptron():
    """E1：感知机在线性可分数据上成功 + 在 XOR 上失败。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    # 左：线性可分
    X, y = make_blobs_2d(n=150, seed=1)
    p = Perceptron(lr=0.1, n_iters=100); p.fit(X, y)
    plot_decision_boundary(ax1, p.predict, X, y, "线性可分：感知机一条线搞定")
    # 右：XOR
    Xx, yx = xor_data()
    px = Perceptron(lr=0.1, n_iters=100); px.fit(Xx, yx)
    ax2.scatter(Xx[:, 0], Xx[:, 1], c=yx, cmap="coolwarm", s=300, edgecolors="k")
    for i in range(len(Xx)):
        ax2.annotate(f"{yx[i]}", Xx[i], fontsize=14, ha="center", va="center")
    ax2.set_title("异或(XOR)：一条直线永远分不开")
    ax2.set_xlim(-0.5, 1.5); ax2.set_ylim(-0.5, 1.5)
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    fig.suptitle("感知机的能与不能：搞得定直线，搞不定异或",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "e1_perceptron.png")


def export_e3_mlp():
    """E3：MLP 训练曲线 + 手写数字预测可视化。"""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X, y = digits.data, digits.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    mlp = MLP(hidden_size=64, lr=0.2, n_epochs=200, seed=0)
    mlp.fit(Xtr, ytr)
    test_acc = np.mean(mlp.predict(Xte) == yte)

    fig = plt.figure(figsize=(13, 4.5))
    # 左：训练曲线
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(mlp.loss_history_, color="#E53935", lw=2, label="训练损失")
    ax1.set_xlabel("训练轮数 epoch"); ax1.set_ylabel("损失", color="#E53935")
    ax2b = ax1.twinx()
    ax2b.plot(mlp.acc_history_, color="#2E7D32", lw=2, label="训练准确率")
    ax2b.set_ylabel("准确率", color="#2E7D32")
    ax1.set_title(f"纯 NumPy 手写 MLP 训练过程（测试准确率 {test_acc:.1%}）")
    # 右：预测可视化（16 张手写数字 + 预测标签）
    ax_grid = fig.add_subplot(1, 2, 2)
    ax_grid.axis("off")
    ax_grid.set_title("模型对手写数字的预测")
    preds = mlp.predict(Xte[:16])
    for i in range(16):
        sub = fig.add_axes([0.55 + (i % 4) * 0.10, 0.55 - (i // 4) * 0.16, 0.09, 0.14])
        sub.imshow(Xte[i].reshape(8, 8), cmap="gray_r")
        color = "green" if preds[i] == yte[i] else "red"
        sub.set_title(f"{preds[i]}", fontsize=11, color=color, pad=1)
        sub.axis("off")
    fig.suptitle("手写 MLP 识别手写数字：训练曲线 + 真实预测",
                 fontsize=15, fontweight="bold", y=1.02)
    return save_fig(fig, OUT, "e3_mlp.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    for fn in (export_e1_perceptron, export_e3_mlp):
        print("  ", fn())


if __name__ == "__main__":
    main()
