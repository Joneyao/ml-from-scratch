import numpy as np
from common import datasets, plotting


def test_make_linear_data_shape():
    X, y = datasets.make_linear_data(n=50, slope=2.0, intercept=1.0, noise=0.0, seed=0)
    assert X.shape == (50,)
    assert y.shape == (50,)
    # 无噪声时应严格落在直线上
    assert np.allclose(y, 2.0 * X + 1.0)


def test_make_poly_data_reproducible():
    X1, y1 = datasets.make_poly_data(n=30, seed=42)
    X2, y2 = datasets.make_poly_data(n=30, seed=42)
    assert np.allclose(X1, X2) and np.allclose(y1, y2)


def test_setup_font_returns_name():
    name = plotting.setup_cjk_font()
    assert isinstance(name, str) and len(name) > 0
