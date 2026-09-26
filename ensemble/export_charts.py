"""运行集成学习 3 个模块，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt

from common.plotting import setup_cjk_font, save_fig
from common.datasets import load_breast_cancer_data
from supervised.decision_tree import DecisionTree
from ensemble.random_forest import RandomForest
from ensemble.gbdt import GBDT
from ensemble import evaluation as ev

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def _split(X, y, ratio=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int(len(X) * ratio)
    tr, te = idx[:k], idx[k:]
    return X[tr], y[tr], X[te], y[te]


def export_c1_forest_vs_tree():
    """C1：单棵树 vs 不同规模森林的测试准确率。"""
    X, y = load_breast_cancer_data()
    Xtr, ytr, Xte, yte = _split(X, y, seed=1)
    # 单棵树（多次平均）
    single = np.mean([
        np.mean(DecisionTree(max_depth=5).fit(*_split(X, y, seed=s)[:2])
                .predict(_split(X, y, seed=s)[2]) == _split(X, y, seed=s)[3])
        for s in range(5)])
    n_trees_list = [1, 5, 10, 20, 40]
    forest_acc = []
    for n in n_trees_list:
        rf = RandomForest(n_trees=n, max_depth=5, seed=1)
        rf.fit(Xtr, ytr)
        forest_acc.append(np.mean(rf.predict(Xte) == yte))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.axhline(single, color="#E53935", ls="--", lw=2, label=f"单棵树 ≈ {single:.3f}")
    ax.plot(n_trees_list, forest_acc, "o-", color="#2E7D32", lw=2, markersize=7,
            label="随机森林")
    ax.set_xlabel("森林里树的棵数"); ax.set_ylabel("测试集准确率")
    ax.set_title("随机森林：树越多，越稳越准"); ax.legend()
    return save_fig(fig, OUT, "c1_forest_vs_tree.png")


def export_c2_gbdt_residual():
    """C2：GBDT 逐轮训练 MSE 下降 + 逐步逼近曲线。"""
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(-3, 3, 80)).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, 0.15, 80)
    gb = GBDT(n_estimators=60, learning_rate=0.2, max_depth=3)
    gb.fit(X, y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    # 左：逐轮 MSE
    ax1.plot(range(1, len(gb.train_loss_) + 1), gb.train_loss_, color="#E53935", lw=2)
    ax1.set_xlabel("树的轮数"); ax1.set_ylabel("训练 MSE")
    ax1.set_title("GBDT：每加一棵树，误差就降一点")
    # 右：不同轮数的拟合
    xs = np.linspace(-3, 3, 200).reshape(-1, 1)
    ax2.scatter(X, y, color="#BBB", s=18, label="数据")
    for n, c in [(1, "#FFCDD2"), (5, "#EF9A9A"), (60, "#B71C1C")]:
        g = GBDT(n_estimators=n, learning_rate=0.2, max_depth=3); g.fit(X, y)
        ax2.plot(xs, g.predict(xs), color=c, lw=2, label=f"{n} 棵树")
    ax2.set_title("树越多，拟合越贴合真实曲线"); ax2.legend(fontsize=9)
    fig.suptitle("GBDT：串行纠错，一棵接一棵补前面的短",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "c2_gbdt_residual.png")


def export_c3_evaluation():
    """C3：混淆矩阵 + ROC 曲线（用随机森林在真实数据上的输出）。"""
    X, y = load_breast_cancer_data()
    Xtr, ytr, Xte, yte = _split(X, y, seed=2)
    rf = RandomForest(n_trees=30, max_depth=5, seed=2)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    cm = ev.confusion_matrix(yte, pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # 左：混淆矩阵热力图
    im = ax1.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=18, color="black")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["预测负", "预测正"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["实际负", "实际正"])
    ax1.set_title(f"混淆矩阵（准确率 {ev.accuracy(yte, pred):.3f}）")
    # 右：ROC
    scores = rf.predict(Xte).astype(float)   # 简化：用预测作为分数
    # 用逐棵树投票比例做连续分数，效果更好
    votes = np.mean([t.predict(Xte[:, f]) for t, f in zip(rf.trees_, rf.features_)], axis=0) \
        if hasattr(rf, "trees_") and hasattr(rf, "features_") else scores
    fpr, tpr, _ = ev.roc_curve(yte, votes)
    ax2.plot(fpr, tpr, color="#E53935", lw=2, label=f"AUC = {ev.auc(fpr, tpr):.3f}")
    ax2.plot([0, 1], [0, 1], color="#999", ls="--")
    ax2.set_xlabel("假正率 FPR"); ax2.set_ylabel("真正率 TPR")
    ax2.set_title("ROC 曲线：越靠左上越好"); ax2.legend()
    fig.suptitle("模型评估：光看准确率不够，还要看混淆矩阵和 ROC",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "c3_evaluation.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    for fn in (export_c1_forest_vs_tree, export_c2_gbdt_residual, export_c3_evaluation):
        print("  ", fn())


if __name__ == "__main__":
    main()
