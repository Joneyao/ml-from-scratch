import numpy as np
from foundations.gradient_descent import gradient, fit
from foundations.linear_fit import mse_loss
from common.datasets import make_linear_data


def test_gradient_sign():
    X, y = make_linear_data(n=30, slope=2.0, intercept=1.0, noise=0.0, seed=0)
    # 从 w 偏小处出发，dw 应为负（提示 w 要增大）
    dw, db = gradient(0.0, 0.0, X, y)
    assert dw < 0


def test_fit_converges():
    X, y = make_linear_data(n=60, slope=2.0, intercept=1.0, noise=0.0, seed=2)
    out = fit(X, y, lr=0.05, n_iters=2000, w0=0.0, b0=0.0)
    assert abs(out["w"] - 2.0) < 0.05 and abs(out["b"] - 1.0) < 0.05


def test_history_records_decreasing_loss():
    X, y = make_linear_data(n=60, slope=2.0, intercept=1.0, noise=0.0, seed=2)
    out = fit(X, y, lr=0.05, n_iters=100, w0=0.0, b0=0.0)
    losses = [h["loss"] for h in out["history"]]
    assert losses[-1] < losses[0]
    assert len(out["history"]) == 100
