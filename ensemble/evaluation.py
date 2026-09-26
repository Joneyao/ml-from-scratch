"""模型评估指标——从零手写混淆矩阵 / 精确率召回率 / ROC / AUC。

针对二分类问题（标签取值 0/1）。核心是先把预测结果和真实标签的四种
组合数清楚（混淆矩阵），再从这四个数派生出各种指标：

    - 准确率 accuracy：预测对的样本占全部样本的比例。直观，但在类别
      极不平衡时会骗人（全预测多数类也能拿到很高的准确率）。
    - 精确率 precision：预测为正的样本里，真正是正类的比例。回答
      "报了这么多警，有几个是真的"。
    - 召回率 recall：真正的正类里，被模型抓出来的比例。回答"该抓的
      抓到了几个"。
    - F1 分数 f1_score：精确率和召回率的调和平均，两者兼顾时才高。
    - ROC 曲线 / AUC：不固定阈值，扫描所有阈值画出 (FPR, TPR) 曲线，
      AUC 是曲线下面积，衡量模型整体的排序能力。

整篇只用 NumPy，不依赖 sklearn。
"""
import numpy as np


def confusion_matrix(y_true, y_pred):
    """计算 2x2 混淆矩阵，返回 [[TN, FP], [FN, TP]]。

    四个格子的含义（约定正类为 1、负类为 0）：
        TN（真负）：真实为 0，预测也为 0。
        FP（假正）：真实为 0，却预测成 1（误报）。
        FN（假负）：真实为 1，却预测成 0（漏报）。
        TP（真正）：真实为 1，预测也为 1。

    行代表真实类别（第 0 行是负类、第 1 行是正类），列代表预测类别
    （第 0 列是负类、第 1 列是正类）。
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred):
    """准确率 = 预测正确的样本数 / 总样本数。

    最直观的指标，但类别不平衡时不可靠：95% 都是负类时，
    一个只会输出负类的模型也能拿到 0.95 的准确率。
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    """精确率 = TP / (TP + FP)。

    在所有"被预测为正类"的样本里，真正是正类的比例。分母为 0
    （即模型一个正类都没预测）时安全返回 0.0。
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(tp / denom)


def recall(y_true, y_pred):
    """召回率 = TP / (TP + FN)。

    在所有"真实为正类"的样本里，被模型正确抓出来的比例。分母为 0
    （即数据里根本没有正类）时安全返回 0.0。
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = tp + fn
    if denom == 0:
        return 0.0
    return float(tp / denom)


def f1_score(y_true, y_pred):
    """F1 = 2 * P * R / (P + R)，精确率与召回率的调和平均。

    只有精确率和召回率都不低时 F1 才高。两者之和为 0 时安全返回 0.0。
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    if denom == 0:
        return 0.0
    return float(2 * p * r / denom)


def roc_curve(y_true, y_score):
    """计算 ROC 曲线，返回 (fpr, tpr, thresholds)。

    ROC 曲线不固定判正阈值，而是把阈值从高到低扫一遍。每取一个阈值，
    就把得分 >= 阈值的样本判为正类，算出这一阈值下的：
        TPR（真正率，即召回率）= TP / (TP + FN)
        FPR（假正率）          = FP / (FP + TN)
    阈值越低，判为正的样本越多，TPR 和 FPR 都单调上升。把每个阈值对应
    的 (FPR, TPR) 连起来就是 ROC 曲线。

    实现思路（等价于 sklearn 的做法）：
        1. 按得分从高到低排序。
        2. 沿排序累加正类数得到 TP、累加负类数得到 FP。
        3. 只在得分发生变化的位置记一个点（同分样本合并成一个阈值）。
        4. 最前面补一个 (0, 0) 点，对应"谁都不判正"的起点。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    # 正类、负类总数，用于把累计计数归一化成 TPR / FPR
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    # 按得分从高到低排序
    order = np.argsort(y_score, kind="mergesort")[::-1]
    scores_sorted = y_score[order]
    labels_sorted = y_true[order]

    # 沿排序方向累计真正例、假正例的数量
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)

    # 只在得分变化处保留阈值点：同分样本必须一起判，不能拆开
    distinct = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct, len(scores_sorted) - 1]

    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    thresholds = scores_sorted[threshold_idxs]

    # 最前面补 (0, 0) 起点：阈值高到没有任何样本被判正
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1.0, thresholds]

    # 归一化。分母为 0（缺正类或缺负类）时该轴退化为全 0，避免除零。
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps, dtype=float)

    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def auc(fpr, tpr):
    """用梯形法计算曲线下面积（AUC）。

    把 ROC 曲线相邻两点之间近似成一个梯形，面积等于
    (x2 - x1) * (y1 + y2) / 2，逐段累加即得曲线下总面积。这正是
    NumPy 的 np.trapz 做的事。AUC 越接近 1，模型排序能力越强；
    0.5 相当于随机猜。
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.size < 2:
        return 0.0
    # 确保横坐标单调递增再做梯形积分
    order = np.argsort(fpr, kind="mergesort")
    return float(np.trapezoid(tpr[order], fpr[order]))
