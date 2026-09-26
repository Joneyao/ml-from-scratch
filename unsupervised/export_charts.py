"""运行无监督学习 3 个模块，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt

from common.plotting import setup_cjk_font, save_fig
from common.datasets import make_blobs_2d
from unsupervised.kmeans import KMeans
from unsupervised.pca import PCA
from unsupervised.anomaly_detection import GaussianAnomalyDetector

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_d1_kmeans():
    """D1：K-Means 聚类结果 + 肘部法则选 K。"""
    X, _ = make_blobs_2d(n=200, seed=5)
    # 强行造 3 簇更直观
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2,
                      random_state=5, cluster_std=1.0)
    km = KMeans(n_clusters=3, seed=0); km.fit(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=25)
    ax1.scatter(km.centroids_[:, 0], km.centroids_[:, 1], c="red",
                marker="*", s=400, edgecolors="k", label="质心")
    ax1.set_title("K-Means：把点自动分成 3 堆，红星是质心"); ax1.legend()
    ax1.set_xticks([]); ax1.set_yticks([])
    # 肘部法则
    ks = range(1, 9)
    inertias = []
    for k in ks:
        m = KMeans(n_clusters=k, seed=0); m.fit(X)
        inertias.append(m.inertia_)
    ax2.plot(list(ks), inertias, "o-", color="#1565C0", lw=2, markersize=7)
    ax2.axvline(3, color="#E53935", ls="--", label="肘部：K=3")
    ax2.set_xlabel("簇的个数 K"); ax2.set_ylabel("簇内平方和")
    ax2.set_title("肘部法则：曲线拐弯处就是好的 K"); ax2.legend()
    fig.suptitle("K-Means 聚类与如何选 K", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, OUT, "d1_kmeans.png")


def export_d2_pca():
    """D2：把 64 维手写数字降到 2 维，看不同数字自动分开。"""
    from sklearn.datasets import load_digits
    d = load_digits()
    X, y = d.data, d.target
    Z = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=15, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="数字 0-9", ticks=range(10))
    ax.set_xlabel("第 1 主成分"); ax.set_ylabel("第 2 主成分")
    ax.set_title("PCA：64 维手写数字压到 2 维，同数字自动聚在一起")
    ax.set_xticks([]); ax.set_yticks([])
    return save_fig(fig, OUT, "d2_pca.png")


def export_d3_anomaly():
    """D3：正常点扎堆 + 离群点被高斯密度检测揪出。"""
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (200, 2))
    outliers = rng.uniform(-9, 9, (12, 2))
    X = np.vstack([normal, outliers])
    det = GaussianAnomalyDetector(contamination=0.06); det.fit(X)
    pred = det.predict(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    normal_mask = pred == 1
    ax.scatter(X[normal_mask, 0], X[normal_mask, 1], c="#1565C0", s=25, label="正常")
    ax.scatter(X[~normal_mask, 0], X[~normal_mask, 1], c="red", marker="x",
               s=120, linewidths=2.5, label="判为异常")
    ax.set_title("异常检测：正常点扎堆，离群点被高斯密度揪出")
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    return save_fig(fig, OUT, "d3_anomaly.png")


def main():
    setup_cjk_font()
    os.makedirs(OUT, exist_ok=True)
    for fn in (export_d1_kmeans, export_d2_pca, export_d3_anomaly):
        print("  ", fn())


if __name__ == "__main__":
    main()
