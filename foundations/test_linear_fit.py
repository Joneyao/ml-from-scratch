import numpy as np
from foundations.linear_fit import mse_loss, grid_search
from common.datasets import make_linear_data


def test_mse_zero_on_perfect_fit():
    X, y = make_linear_data(n=20, slope=2.0, intercept=1.0, noise=0.0, seed=0)
    assert mse_loss(2.0, 1.0, X, y) < 1e-9


def test_mse_positive_when_wrong():
    X, y = make_linear_data(n=20, slope=2.0, intercept=1.0, noise=0.0, seed=0)
    assert mse_loss(0.0, 0.0, X, y) > 1.0


def test_grid_search_recovers_params():
    X, y = make_linear_data(n=40, slope=2.0, intercept=1.0, noise=0.0, seed=1)
    w, b, grid = grid_search(X, y, np.linspace(0, 4, 81), np.linspace(-1, 3, 81))
    assert abs(w - 2.0) < 0.1 and abs(b - 1.0) < 0.1
    assert grid.shape == (81, 81)
