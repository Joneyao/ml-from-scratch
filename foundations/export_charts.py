"""运行地基篇模块，导出真实数据配图（PNG + CSV）到 charts_out/。

所有图的数据都来自模块真实运行，不写死数字。文章直接引用这些 PNG。
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from common.plotting import setup_cjk_font, save_fig
from common.datasets import make_linear_data, make_poly_data
from foundations import linear_fit, gradient_descent, overfitting

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


# ---------- A1：损失与拟合 ----------

def export_a1_loss_surface():
    """损失曲面等高线：不同 (w,b) 下 MSE 的高低，最低点就是最优参数。"""
    X, y = make_linear_data(n=40, slope=2.0, intercept=1.0, noise=0.5, seed=1)
    w_range = np.linspace(0, 4, 81)
    b_range = np.linspace(-1, 3, 81)
    bw, bb, grid = linear_fit.grid_search(X, y, w_range, b_range)
    fig, ax = plt.subplots(figsize=(6, 5))
    W, B = np.meshgrid(w_range, b_range, indexing="ij")
    cs = ax.contourf(W, B, grid, levels=30, cmap="viridis")
    ax.plot(bw, bb, "r*", markersize=18, label=f"最优点 w={bw:.2f}, b={bb:.2f}")
    fig.colorbar(cs, ax=ax, label="损失（MSE）")
    ax.set_xlabel("斜率 w"); ax.set_ylabel("截距 b")
    ax.set_title("损失曲面：找最低点就是训练"); ax.legend()
    return save_fig(fig, OUT, "loss_surface.png")


def export_a1_loss_vs_w():
    """固定最优 b，损失随斜率 w 的变化曲线——一个碗形。"""
    X, y = make_linear_data(n=40, slope=2.0, intercept=1.0, noise=0.5, seed=1)
    ws = np.linspace(-1, 5, 121)
    losses = [linear_fit.mse_loss(w, 1.0, X, y) for w in ws]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, losses, lw=2)
    best_w = ws[int(np.argmin(losses))]
    ax.axvline(best_w, color="r", ls="--", label=f"最低点 w≈{best_w:.2f}")
    ax.set_xlabel("斜率 w"); ax.set_ylabel("损失（MSE）")
    ax.set_title("损失是一个碗——底部就是答案"); ax.legend()
    return save_fig(fig, OUT, "loss_vs_w.png")


# ---------- A2：梯度下降 ----------

def export_a2_lr_comparison():
    """不同学习率的收敛对比：太小慢、合适快、太大震荡。"""
    X, y = make_linear_data(n=60, slope=2.0, intercept=1.0, noise=0.5, seed=2)
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr, label in [(0.01, "太小 lr=0.01"), (0.1, "合适 lr=0.1"), (0.9, "太大 lr=0.9")]:
        out = gradient_descent.fit(X, y, lr=lr, n_iters=60, w0=0.0, b0=0.0)
        losses = [h["loss"] for h in out["history"]]
        ax.plot(losses, label=label)
    ax.set_xlabel("迭代轮数"); ax.set_ylabel("损失")
    ax.set_title("学习率对收敛的影响"); ax.legend(); ax.set_yscale("log")
    return save_fig(fig, OUT, "lr_comparison.png")


def export_a2_trajectory():
    """梯度下降在损失等高线上的下降轨迹——一步步走向最低点。"""
    X, y = make_linear_data(n=60, slope=2.0, intercept=1.0, noise=0.5, seed=2)
    out = gradient_descent.fit(X, y, lr=0.1, n_iters=40, w0=0.0, b0=0.0)
    ws = np.array([h["w"] for h in out["history"]])
    bs = np.array([h["b"] for h in out["history"]])
    w_range = np.linspace(-0.5, 3, 80)
    b_range = np.linspace(-0.5, 2.5, 80)
    _, _, grid = linear_fit.grid_search(X, y, w_range, b_range)
    fig, ax = plt.subplots(figsize=(6, 5))
    W, B = np.meshgrid(w_range, b_range, indexing="ij")
    ax.contour(W, B, grid, levels=25, cmap="viridis", alpha=0.6)
    ax.plot(ws, bs, "o-", color="red", markersize=3, lw=1, label="下降轨迹")
    ax.plot(ws[0], bs[0], "ks", markersize=8, label="起点")
    ax.plot(ws[-1], bs[-1], "r*", markersize=16, label="终点")
    ax.set_xlabel("斜率 w"); ax.set_ylabel("截距 b")
    ax.set_title("梯度下降：像蒙眼下山一步步走向谷底"); ax.legend()
    return save_fig(fig, OUT, "gd_trajectory.png")


def export_a2_gd_vs_grid():
    """梯度下降 vs 暴力网格搜索：达到同样损失所需的评估次数对比。"""
    X, y = make_linear_data(n=60, slope=2.0, intercept=1.0, noise=0.5, seed=2)
    out = gradient_descent.fit(X, y, lr=0.1, n_iters=50, w0=0.0, b0=0.0)
    gd_final = out["history"][-1]["loss"]
    gd_evals = len(out["history"])                 # 每轮 1 次评估量级
    grid_n = 81
    grid_evals = grid_n * grid_n                    # 网格搜索评估次数
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["梯度下降", "暴力网格搜索"], [gd_evals, grid_evals],
                  color=["#2196F3", "#FF9800"])
    ax.bar_label(bars)
    ax.set_ylabel("参数评估次数（越少越快）")
    ax.set_title(f"两种找最优的代价对比（最终损失≈{gd_final:.3f}）")
    return save_fig(fig, OUT, "gd_vs_grid.png")


# ---------- A3：过拟合与正则化 ----------

def export_a3_train_test():
    """不同阶数的训练/测试误差分叉曲线——过拟合的经典信号。"""
    os.makedirs(OUT, exist_ok=True)
    degrees = list(range(1, 15))
    tr_list, te_list = [], []
    for d in degrees:
        tr, te = overfitting.train_test_errors(degree=d, alpha=0.0, seed=0)
        tr_list.append(tr); te_list.append(te)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(degrees, tr_list, "o-", label="训练误差")
    ax.plot(degrees, te_list, "s-", label="测试误差")
    ax.set_xlabel("多项式阶数"); ax.set_ylabel("均方误差")
    ax.set_title("过拟合：训练误差和测试误差的分叉"); ax.legend(); ax.set_yscale("log")
    with open(os.path.join(OUT, "train_test.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["degree", "train_mse", "test_mse"])
        w.writerows(zip(degrees, tr_list, te_list))
    return save_fig(fig, OUT, "train_test_split.png")


def export_a3_fit_curves():
    """欠拟合/合适/过拟合三条拟合曲线并排——直观看'背答案'长什么样。"""
    Xtr, ytr = make_poly_data(n=25, seed=0)
    xs = np.linspace(-1, 1, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    configs = [(1, 0.0, "欠拟合（1 阶）"), (3, 0.0, "合适（3 阶）"),
               (14, 0.0, "过拟合（14 阶）")]
    for ax, (deg, alpha, title) in zip(axes, configs):
        coef, bias = overfitting.fit_ridge(overfitting.poly_features(Xtr, deg), ytr, alpha)
        ax.scatter(Xtr, ytr, s=20, color="#555", label="训练点")
        ax.plot(xs, overfitting.predict_poly(coef, bias, xs, deg), "r", lw=2, label="拟合曲线")
        ax.set_title(title); ax.set_ylim(-3, 3); ax.legend(fontsize=8)
    fig.suptitle("从欠拟合到过拟合：模型复杂度的三种状态")
    return save_fig(fig, OUT, "fit_curves.png")


def export_a3_reg_effect():
    """同为高阶模型，加 L2 正则前后的拟合对比——正则把'扭曲'压平。"""
    Xtr, ytr = make_poly_data(n=25, seed=0)
    xs = np.linspace(-1, 1, 200)
    deg = 14
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (alpha, title) in zip(axes, [(0.0, "无正则（过拟合）"), (1.0, "L2 正则 α=1.0")]):
        coef, bias = overfitting.fit_ridge(overfitting.poly_features(Xtr, deg), ytr, alpha)
        ax.scatter(Xtr, ytr, s=20, color="#555", label="训练点")
        ax.plot(xs, overfitting.predict_poly(coef, bias, xs, deg), "r", lw=2, label="拟合曲线")
        ax.set_title(title); ax.set_ylim(-3, 3); ax.legend(fontsize=8)
    fig.suptitle("正则化：给'背答案'的模型踩一脚刹车")
    return save_fig(fig, OUT, "reg_effect.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    paths = [
        export_a1_loss_surface(),
        export_a1_loss_vs_w(),
        export_a2_lr_comparison(),
        export_a2_trajectory(),
        export_a2_gd_vs_grid(),
        export_a3_train_test(),
        export_a3_fit_curves(),
        export_a3_reg_effect(),
    ]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
