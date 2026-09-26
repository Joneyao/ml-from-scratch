"""运行监督学习 6 个模块，导出真实数据配图（PNG）到 charts_out/。

图数据全部来自模块真实运行，供第二批文章直接引用。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from common.plotting import setup_cjk_font, save_fig, plot_decision_boundary
from common.datasets import (
    make_linear_data, make_blobs_2d, make_two_moons, load_iris_binary,
)
from supervised.linear_regression import LinearRegression
from supervised.logistic_regression import LogisticRegression
from supervised.knn import KNN
from supervised.naive_bayes import GaussianNB
from supervised.decision_tree import DecisionTree
from supervised.svm import LinearSVM, rbf_kernel

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_b1_linear():
    """B1：线性回归拟合直线。"""
    X, y = make_linear_data(n=40, slope=2.0, intercept=1.0, noise=1.2, seed=1)
    reg = LinearRegression(); reg.fit(X, y)
    xs = np.linspace(X.min(), X.max(), 100)
    ys = reg.predict(xs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(X, y, color="#555", s=30, label="数据点")
    ax.plot(xs, ys, color="#E53935", lw=2.5, label="拟合直线")
    ax.set_title("线性回归：找一条最贴合数据的直线")
    ax.set_xlabel("特征 x"); ax.set_ylabel("目标 y"); ax.legend()
    return save_fig(fig, OUT, "b1_linear_fit.png")


def export_b2_logistic():
    """B2：逻辑回归决策边界。"""
    X, y = make_blobs_2d(n=200, seed=1)
    clf = LogisticRegression(); clf.fit(X, y)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    plot_decision_boundary(ax, clf.predict, X, y, "逻辑回归：一条直线分开两类")
    ax.set_xlabel("特征 1"); ax.set_ylabel("特征 2")
    return save_fig(fig, OUT, "b2_logistic_boundary.png")


def export_b3_knn():
    """B3：不同 K 值的决策边界（K 越大越平滑）。"""
    X, y = make_blobs_2d(n=150, seed=4)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, k in zip(axes, [1, 5, 25]):
        clf = KNN(k=k); clf.fit(X, y)
        plot_decision_boundary(ax, clf.predict, X, y, f"K = {k}")
    fig.suptitle("KNN：K 越大，决策边界越平滑", fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    return save_fig(fig, OUT, "b3_knn_kvalues.png")


def export_b4_naive_bayes():
    """B4：高斯朴素贝叶斯决策边界。"""
    X, y = load_iris_binary()
    clf = GaussianNB(); clf.fit(X, y)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    plot_decision_boundary(ax, clf.predict, X, y, "高斯朴素贝叶斯：用概率划分鸢尾花")
    ax.set_xlabel("花萼长度"); ax.set_ylabel("花萼宽度")
    return save_fig(fig, OUT, "b4_naive_bayes_boundary.png")


def export_b5_decision_tree():
    """B5：不同深度决策树的阶梯状边界。"""
    X, y = make_blobs_2d(n=150, seed=4)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, depth in zip(axes, [1, 3, 8]):
        clf = DecisionTree(max_depth=depth); clf.fit(X, y)
        plot_decision_boundary(ax, clf.predict, X, y, f"最大深度 = {depth}")
    fig.suptitle("决策树：深度越大，边界越细碎（越容易过拟合）",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    return save_fig(fig, OUT, "b5_tree_depth.png")


def export_b6_svm():
    """B6：线性 SVM 边界 + RBF 核在月牙数据上的非线性边界。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # 左：线性 SVM
    X1, y1 = make_blobs_2d(n=150, seed=1)
    svm = LinearSVM(); svm.fit(X1, y1, lr=0.01, n_iters=2000, C=1.0)
    plot_decision_boundary(axes[0], svm.predict, X1, y1, "线性 SVM：最宽的分隔带")
    # 右：RBF 核 SVM（用核化最近类中心思路演示非线性可分）
    X2, y2 = make_two_moons(n=200, noise=0.15, seed=0)
    _rbf_boundary(axes[1], X2, y2)
    axes[1].set_title("RBF 核：把线性分不开的月牙掰弯")
    fig.suptitle("SVM：从直线间隔到核技巧", fontsize=15, fontweight="bold")
    return save_fig(fig, OUT, "b6_svm.png")


def _rbf_boundary(ax, X, y, gamma=2.0):
    """用 RBF 核相似度做最近类中心分类，演示核技巧的非线性边界。"""
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 每个网格点对两类样本的核相似度之和，取更相似的类
    sim0 = rbf_kernel(grid, X[y == 0], gamma).sum(axis=1) / np.sum(y == 0)
    sim1 = rbf_kernel(grid, X[y == 1], gamma).sum(axis=1) / np.sum(y == 1)
    Z = (sim1 > sim0).astype(int).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=22)


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    paths = [
        export_b1_linear(), export_b2_logistic(), export_b3_knn(),
        export_b4_naive_bayes(), export_b5_decision_tree(), export_b6_svm(),
    ]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
