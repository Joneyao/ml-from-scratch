"""过拟合与 L2 正则化——多项式拟合，正规方程 + 岭回归闭式解。"""
import numpy as np
from common.datasets import make_poly_data


def poly_features(X, degree):
    """构造 [x, x^2, ..., x^degree] 特征矩阵（不含偏置列）。"""
    return np.vstack([X ** d for d in range(1, degree + 1)]).T


def fit_ridge(Phi, y, alpha):
    """岭回归闭式解，返回 (coef, bias)。偏置不参与正则。"""
    n, d = Phi.shape
    Xb = np.hstack([np.ones((n, 1)), Phi])          # 加偏置列
    reg = alpha * np.eye(d + 1)
    reg[0, 0] = 0.0                                  # 偏置项不正则
    theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return theta[1:], theta[0]


def predict_poly(coef, bias, X, degree):
    return poly_features(X, degree) @ coef + bias


def train_test_errors(degree, alpha, seed=0):
    Xtr, ytr = make_poly_data(n=25, seed=seed)
    Xte, yte = make_poly_data(n=25, seed=seed + 100)
    coef, bias = fit_ridge(poly_features(Xtr, degree), ytr, alpha)
    tr = float(np.mean((predict_poly(coef, bias, Xtr, degree) - ytr) ** 2))
    te = float(np.mean((predict_poly(coef, bias, Xte, degree) - yte) ** 2))
    return tr, te
