"""机器学习三件套——模型 y=wx+b、MSE 损失、暴力网格搜参数。"""
import numpy as np


def predict(w, b, X):
    return w * X + b


def mse_loss(w, b, X, y):
    """均方误差：预测值和真实值差的平方的平均。"""
    err = predict(w, b, X) - y
    return float(np.mean(err ** 2))


def grid_search(X, y, w_range, b_range):
    """在参数网格上暴力搜索最小损失的 (w, b)。返回最优参数和损失网格。"""
    loss_grid = np.empty((len(w_range), len(b_range)))
    for i, w in enumerate(w_range):
        for j, b in enumerate(b_range):
            loss_grid[i, j] = mse_loss(w, b, X, y)
    i_best, j_best = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    return float(w_range[i_best]), float(b_range[j_best]), loss_grid
