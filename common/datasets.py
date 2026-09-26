"""教学用数据集生成。"""
import numpy as np


def make_linear_data(n=50, slope=2.0, intercept=1.0, noise=1.0, seed=0):
    """生成一条带噪声的直线数据 y = slope*x + intercept + 噪声。"""
    rng = np.random.default_rng(seed)
    X = np.linspace(-3, 3, n)
    y = slope * X + intercept + rng.normal(0, noise, size=n)
    return X, y


def make_poly_data(n=30, noise=0.3, seed=0):
    """真实关系是三次多项式，用来演示欠拟合/过拟合。"""
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-1, 1, size=n))
    y_true = 1.0 - 2.0 * X + 0.5 * X**2 + 1.5 * X**3
    y = y_true + rng.normal(0, noise, size=n)
    return X, y


def make_blobs_2d(n=100, seed=0):
    """两簇二维高斯点，用于画决策边界。"""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, centers=2, n_features=2,
                      random_state=seed, cluster_std=1.5)
    return X, y


def make_two_moons(n=200, noise=0.2, seed=0):
    """两个月牙形，线性不可分，用于展示核方法/非线性。"""
    from sklearn.datasets import make_moons
    return make_moons(n_samples=n, noise=noise, random_state=seed)


def load_iris_binary():
    """鸢尾花，取前两类 + 前两个特征，便于二分类可视化。"""
    from sklearn.datasets import load_iris
    d = load_iris()
    mask = d.target < 2
    return d.data[mask][:, :2], d.target[mask]


def load_breast_cancer_data():
    """乳腺癌数据集（30 维特征，二分类）。"""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    return d.data, d.target
