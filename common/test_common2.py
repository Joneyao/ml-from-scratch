import numpy as np
from common import datasets


def test_make_blobs_2d_shape():
    X, y = datasets.make_blobs_2d(n=60, seed=0)
    assert X.shape == (60, 2)
    assert set(np.unique(y)) <= {0, 1}


def test_make_two_moons_shape():
    X, y = datasets.make_two_moons(n=120, seed=0)
    assert X.shape == (120, 2)
    assert set(np.unique(y)) == {0, 1}


def test_load_iris_binary():
    X, y = datasets.load_iris_binary()
    assert X.shape[1] == 2          # 只取 2 个特征便于可视化
    assert set(np.unique(y)) == {0, 1}
