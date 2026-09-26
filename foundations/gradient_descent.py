"""梯度下降——沿损失下降最快的方向一步步更新参数。"""
import numpy as np
from foundations.linear_fit import mse_loss


def gradient(w, b, X, y):
    """MSE 对 w、b 的偏导数。"""
    err = (w * X + b) - y
    dw = float(2.0 * np.mean(err * X))
    db = float(2.0 * np.mean(err))
    return dw, db


def fit(X, y, lr=0.05, n_iters=1000, w0=0.0, b0=0.0):
    """梯度下降训练，记录每轮轨迹供绘图。"""
    w, b = w0, b0
    history = []
    for _ in range(n_iters):
        dw, db = gradient(w, b, X, y)
        w -= lr * dw
        b -= lr * db
        history.append({"w": w, "b": b, "loss": mse_loss(w, b, X, y)})
    return {"w": w, "b": b, "history": history}
