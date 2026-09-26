"""异常检测的测试：手造离群点，验证两种检测器能把它们揪出来。

异常检测的直觉很朴素：正常点扎堆、离群点孤零零。所以这里的手造数据
都是"一大团正常点 + 几个明显偏离的离群点"，好检验器应该把离群点判成
异常（-1），把正常点判成正常（1）。第三条用 sklearn 的 IsolationForest
做数值对照，确认我方检出率不落下风。
"""
import numpy as np

from unsupervised.anomaly_detection import GaussianAnomalyDetector, ZScoreDetector


def _make_data(n_normal=200, n_outliers=10, seed=0):
    """造一团二维正常点（原点附近的高斯团）+ 一撮明显离群点。

    返回 (X, is_outlier)：X 形状 (n_normal+n_outliers, 2)，is_outlier 是
    布尔数组，True 表示这一行是人为注入的离群点。离群点撒在距离原点
    很远的地方（±8~12），和正常团（std=1）拉开明显距离。
    """
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 2))
    # 离群点：坐标绝对值在 8~12 之间，正负随机，远离正常团
    offset = rng.uniform(8.0, 12.0, size=(n_outliers, 2))
    signs = rng.choice([-1.0, 1.0], size=(n_outliers, 2))
    outliers = offset * signs
    X = np.vstack([normal, outliers])
    is_outlier = np.zeros(X.shape[0], dtype=bool)
    is_outlier[n_normal:] = True
    return X, is_outlier


def test_gaussian_flags_outliers_and_keeps_normal():
    """GaussianAnomalyDetector 应把离群点判为异常(-1)、正常点判为正常(1)。

    离群点在高斯密度下概率极低，必然落在阈值之下。这里要求注入的离群点
    全部被抓到，且正常点里被误报的比例很低（< 15%，容忍少量边缘点）。
    """
    X, is_outlier = _make_data(seed=0)
    det = GaussianAnomalyDetector(contamination=0.05)
    det.fit(X)
    pred = det.predict(X)

    # 注入的离群点应全部判为异常
    assert np.all(pred[is_outlier] == -1)
    # 正常点应大多判为正常，误报率控制在 15% 以内
    normal_pred = pred[~is_outlier]
    false_positive_rate = np.mean(normal_pred == -1)
    assert false_positive_rate < 0.15
    # 阈值属性存在
    assert det.threshold_ is not None


def test_zscore_flags_points_beyond_3_sigma():
    """ZScoreDetector 对一维数据里超过 3 倍标准差的点判为异常。

    造一列均值 0、标准差约 1 的正常数据，再手动塞进几个 z 远大于 3 的点
    （值 = 10、-10）。这些点必被判 -1，而分布中心的点判 1。
    """
    rng = np.random.default_rng(1)
    normal = rng.normal(0.0, 1.0, size=(300, 1))
    outliers = np.array([[10.0], [-10.0], [12.0]])
    X = np.vstack([normal, outliers])

    det = ZScoreDetector(k=3.0)
    det.fit(X)
    pred = det.predict(X)

    # 最后三个是极端离群点，必被判异常
    assert np.all(pred[-3:] == -1)
    # 分布中心的点（原点附近）应判正常：取一个接近均值的样本
    center = np.array([[0.0]])
    assert det.predict(center)[0] == 1


def test_recall_matches_isolation_forest():
    """与 sklearn.ensemble.IsolationForest 对照：我方对注入离群点的检出率不低于 0.8。

    在同一手造数据上，IsolationForest 也应能识别出注入的离群点。这里主要
    校验我方 GaussianAnomalyDetector 的 recall（真离群点被判异常的比例）
    >= 0.8，并确认它识别出的异常集合和 IsolationForest 有明显重叠。
    """
    from sklearn.ensemble import IsolationForest

    X, is_outlier = _make_data(n_normal=300, n_outliers=15, seed=2)

    det = GaussianAnomalyDetector(contamination=0.05)
    det.fit(X)
    my_pred = det.predict(X)

    # 我方 recall：真离群点里被判为异常的比例
    my_recall = np.mean(my_pred[is_outlier] == -1)
    assert my_recall >= 0.8

    # sklearn 对照
    iso = IsolationForest(contamination=0.05, random_state=0)
    iso_pred = iso.fit_predict(X)
    iso_recall = np.mean(iso_pred[is_outlier] == -1)
    assert iso_recall >= 0.8

    # 两者识别出的异常集合应有明显重叠（Jaccard 相似度 > 0.5）
    my_anom = set(np.where(my_pred == -1)[0])
    iso_anom = set(np.where(iso_pred == -1)[0])
    inter = len(my_anom & iso_anom)
    union = len(my_anom | iso_anom)
    assert union > 0
    assert inter / union > 0.5
