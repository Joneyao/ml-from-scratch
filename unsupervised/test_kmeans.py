import numpy as np
from unsupervised.kmeans import KMeans
from common.datasets import make_blobs_2d


def _cluster_accuracy(labels, y_true):
    """两簇场景下把聚类标签和真实标签对齐后的准确率。

    K-Means 给出的簇编号是任意的（簇 0/1 可能和真实类 0/1 反过来），
    所以取 acc 和 1-acc 的较大者作为对齐后的准确率。
    """
    acc = np.mean(labels == y_true)
    return max(acc, 1.0 - acc)


def test_cluster_purity_above_threshold():
    """在两簇（可分）blobs 数据上聚成 2 类，与真实标签对齐后准确率应 > 0.9。

    注意 seed=0 的两簇 std=1.5 有明显重叠，即便最优 K-Means（含 sklearn）
    对齐后也只有 0.89——那是数据本身不可分，不是算法的错。这里选一个
    两簇分得开的 seed，用来验证"簇分得开时点确实聚到一起"这个性质。
    """
    X, y = make_blobs_2d(n=200, seed=1)
    km = KMeans(n_clusters=2, seed=0)
    labels = km.fit_predict(X)
    assert _cluster_accuracy(labels, y) > 0.9


def test_inertia_positive_and_history_decreases():
    """inertia_ 应为正；且随迭代记录的簇内平方和单调不增（收敛）。"""
    X, _ = make_blobs_2d(n=200, seed=1)
    km = KMeans(n_clusters=2, seed=0)
    km.fit(X)
    assert km.inertia_ > 0
    # history_ 记录每轮质心；用它重算每轮 inertia，应单调不增
    inertias = []
    for centroids in km.history_:
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        inertias.append(np.sum(np.min(d, axis=1) ** 2))
    diffs = np.diff(inertias)
    assert np.all(diffs <= 1e-6)


def test_inertia_close_to_sklearn():
    """与 sklearn.cluster.KMeans 的 inertia 数量级一致（相对差 < 20%）。"""
    from sklearn.cluster import KMeans as SKKMeans

    X, _ = make_blobs_2d(n=200, seed=2)
    km = KMeans(n_clusters=2, seed=0)
    km.fit(X)

    sk = SKKMeans(n_clusters=2, n_init=10, random_state=0)
    sk.fit(X)

    rel_diff = abs(km.inertia_ - sk.inertia_) / sk.inertia_
    assert rel_diff < 0.20
