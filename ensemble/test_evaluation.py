import numpy as np

from ensemble.evaluation import (
    confusion_matrix,
    accuracy,
    precision,
    recall,
    f1_score,
    roc_curve,
    auc,
)


def test_handcrafted_confusion_matrix_prf():
    """手造一个混淆矩阵例子，验证 precision/recall/f1 数值正确。

    设计标签使得 TP=3, FP=1, FN=2, TN=4：
      - precision = TP/(TP+FP) = 3/4 = 0.75
      - recall    = TP/(TP+FN) = 3/5 = 0.6
      - f1        = 2PR/(P+R)  = 2*0.75*0.6/(0.75+0.6) = 0.666...
    """
    # 前 5 个真实为正类(1)，后 5 个真实为负类(0)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # 正类里预测对 3 个(TP=3)、漏掉 2 个(FN=2)；负类里错判 1 个(FP=1)、对 4 个(TN=4)
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])

    cm = confusion_matrix(y_true, y_pred)
    # 约定顺序 [[TN, FP], [FN, TP]]
    assert cm.tolist() == [[4, 1], [2, 3]]

    assert abs(precision(y_true, y_pred) - 0.75) < 1e-12
    assert abs(recall(y_true, y_pred) - 0.6) < 1e-12
    assert abs(f1_score(y_true, y_pred) - (2 * 0.75 * 0.6 / (0.75 + 0.6))) < 1e-12


def test_perfect_prediction_accuracy_one():
    """全对预测时 accuracy 应等于 1.0。"""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = y_true.copy()
    assert accuracy(y_true, y_pred) == 1.0


def test_imbalanced_accuracy_lies():
    """类别极不平衡时，全预测多数类：accuracy 很高但 recall==0。

    95 个负类、5 个正类，模型偷懒全预测 0：
      - accuracy = 95/100 = 0.95（看起来很棒）
      - recall   = TP/(TP+FN) = 0/5 = 0.0（正类一个都没抓到）
    这就是"准确率会骗人"——不平衡数据下单看准确率会被误导。
    """
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.zeros(100, dtype=int)  # 全预测负类

    assert abs(accuracy(y_true, y_pred) - 0.95) < 1e-12
    assert recall(y_true, y_pred) == 0.0
    # 分母为 0 的边界：没有任何正类预测，precision 应安全返回 0.0
    assert precision(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0


def test_matches_sklearn_prf_and_auc():
    """与 sklearn.metrics 对照：precision/recall/f1/auc 数值接近。"""
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score as sk_f1,
        roc_auc_score,
    )

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    y_pred = (y_score >= 0.5).astype(int)

    assert abs(precision(y_true, y_pred) - precision_score(y_true, y_pred)) < 1e-6
    assert abs(recall(y_true, y_pred) - recall_score(y_true, y_pred)) < 1e-6
    assert abs(f1_score(y_true, y_pred) - sk_f1(y_true, y_pred)) < 1e-6

    fpr, tpr, _ = roc_curve(y_true, y_score)
    our_auc = auc(fpr, tpr)
    assert abs(our_auc - roc_auc_score(y_true, y_score)) < 0.01
