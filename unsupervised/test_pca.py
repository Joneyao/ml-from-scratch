import numpy as np
from sklearn.datasets import load_digits

from unsupervised.pca import PCA


def test_transform_output_shape():
    """把 64 维的 digits 数据降到 2 维，transform 输出形状应为 (n, 2)。"""
    X = load_digits().data  # (1797, 64)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    assert Z.shape == (X.shape[0], 2)


def test_explained_variance_ratio_sorted_and_bounded():
    """方差解释比例应降序排列，且总和 <= 1。

    每个主成分解释的方差占全体方差的比例都是非负数，按特征值从大到小
    取主成分，比例自然降序；只取前 n_components 个，累计比例不超过 1。
    """
    X = load_digits().data
    pca = PCA(n_components=5)
    pca.fit(X)
    ratio = pca.explained_variance_ratio_
    # 降序：后一个不大于前一个
    assert np.all(np.diff(ratio) <= 1e-12)
    # 每个比例非负，总和不超过 1
    assert np.all(ratio >= -1e-12)
    assert ratio.sum() <= 1.0 + 1e-9


def test_matches_sklearn_variance_ratio():
    """与 sklearn.decomposition.PCA 对照：各主成分方差解释比例接近。

    特征向量的符号可能相反（方向不唯一），因此不直接比较主成分向量，
    只比较每个主成分解释的方差比例，差异应小于 0.01。
    """
    from sklearn.decomposition import PCA as SKPCA

    X = load_digits().data
    n = 5

    ours = PCA(n_components=n)
    ours.fit(X)

    sk = SKPCA(n_components=n)
    sk.fit(X)

    diff = np.abs(ours.explained_variance_ratio_ - sk.explained_variance_ratio_)
    assert np.all(diff < 0.01)
