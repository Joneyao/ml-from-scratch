import numpy as np
from ensemble.gbdt import GBDT, _RegressionTree


def _make_sine_data(n=200, noise=0.1, seed=0):
    """构造一维非线性回归数据 y = sin(x) + 噪声。"""
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0, 2 * np.pi, size=n)).reshape(-1, 1)
    y = np.sin(X[:, 0]) + rng.normal(0, noise, size=n)
    return X, y


def test_regression_tree_deeper_fits_better():
    """回归树能拟合简单数据，且深度越大训练误差越小（拟合能力更强）。"""
    X, y = _make_sine_data(n=150, noise=0.05, seed=1)

    mse_by_depth = []
    for depth in (1, 3, 6):
        tree = _RegressionTree(max_depth=depth)
        tree.fit(X, y)
        pred = tree.predict(X)
        mse_by_depth.append(np.mean((pred - y) ** 2))

    # 深度越大，训练 MSE 单调下降（拟合越充分）
    assert mse_by_depth[0] > mse_by_depth[1] > mse_by_depth[2]


def test_gbdt_train_mse_decreases_with_estimators():
    """GBDT 在非线性数据上，随着树数量增加，训练 MSE 逐轮下降。"""
    X, y = _make_sine_data(n=200, noise=0.1, seed=2)

    model = GBDT(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    loss = model.train_loss_
    # 记录了每轮训练 MSE
    assert len(loss) == 50
    # 最终 MSE 明显低于初始 MSE（初始预测=均值）
    assert loss[-1] < loss[0]
    # 整体趋势下降：后半段平均 MSE 低于前半段
    assert np.mean(loss[25:]) < np.mean(loss[:25])


def test_gbdt_mse_close_to_sklearn():
    """与 sklearn.GradientBoostingRegressor 在同一数据上的预测 MSE 宽松对照。"""
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = _make_sine_data(n=200, noise=0.1, seed=3)

    ours = GBDT(n_estimators=50, learning_rate=0.1, max_depth=3)
    ours.fit(X, y)
    mse_ours = np.mean((ours.predict(X) - y) ** 2)

    sk = GradientBoostingRegressor(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0
    )
    sk.fit(X, y)
    mse_sk = np.mean((sk.predict(X) - y) ** 2)

    # 宽松对照：我方 MSE 不超过 sklearn 的 2 倍
    assert mse_ours < 2.0 * mse_sk
