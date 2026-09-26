import numpy as np
from supervised.linear_regression import LinearRegression
from common.datasets import make_linear_data


def test_normal_equation_recovers_line():
    """无噪声一维直线数据：正规方程解出的斜率/截距应接近真值。"""
    X, y = make_linear_data(n=50, slope=2.0, intercept=1.0, noise=0.0, seed=0)
    model = LinearRegression()
    model.fit(X, y)  # 默认走正规方程
    assert abs(float(model.coef_[0]) - 2.0) < 1e-6
    assert abs(model.intercept_ - 1.0) < 1e-6


def test_gradient_descent_matches_normal_equation():
    """梯度下降解应收敛到接近正规方程的解。"""
    X, y = make_linear_data(n=60, slope=2.0, intercept=1.0, noise=0.5, seed=1)

    m_normal = LinearRegression()
    m_normal.fit_normal_equation(X, y)

    m_gd = LinearRegression()
    m_gd.fit_gradient_descent(X, y, lr=0.05, n_iters=20000)

    assert abs(float(m_gd.coef_[0]) - float(m_normal.coef_[0])) < 1e-2
    assert abs(m_gd.intercept_ - m_normal.intercept_) < 1e-2


def test_matches_sklearn_multivariate():
    """多元数据上与 sklearn 的系数对照，差 < 0.01。"""
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 4
    X = rng.normal(0, 1, size=(n_samples, n_features))
    true_coef = np.array([1.5, -2.0, 0.7, 3.2])
    true_intercept = 4.0
    y = X @ true_coef + true_intercept + rng.normal(0, 0.1, size=n_samples)

    mine = LinearRegression()
    mine.fit(X, y)

    sk = SkLinearRegression()
    sk.fit(X, y)

    assert np.max(np.abs(mine.coef_ - sk.coef_)) < 0.01
    assert abs(mine.intercept_ - sk.intercept_) < 0.01


def test_predict_shape_and_reshape():
    """predict 接受一维输入并返回与样本数一致的一维预测。"""
    X, y = make_linear_data(n=30, slope=2.0, intercept=1.0, noise=0.0, seed=2)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (30,)
    assert np.allclose(y_pred, y, atol=1e-6)
