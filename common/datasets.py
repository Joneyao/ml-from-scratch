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
