"""线性回归——从零手写，提供正规方程闭式解与梯度下降两种解法。

只用 NumPy 实现，不依赖 sklearn。命名（coef_/intercept_/fit/predict）
对齐 sklearn 习惯，方便和 sklearn 做数值对照。
"""
import numpy as np


class LinearRegression:
    """普通最小二乘线性回归 y = X·coef_ + intercept_。

    属性
    ----
    coef_ : np.ndarray, 形状 (n_features,)
        每个特征的系数（权重向量）。
    intercept_ : float
        偏置项（截距）。
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _as_2d(X):
        """把输入统一成二维矩阵 (n_samples, n_features)。

        一维数组视为单特征，reshape 成 (n_samples, 1)。
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _add_bias(X):
        """在特征矩阵最左边拼一列全 1，作为偏置对应的常数项。"""
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        """训练模型。默认使用正规方程闭式解。"""
        return self.fit_normal_equation(X, y)

    def fit_normal_equation(self, X, y):
        """正规方程闭式解： theta = (XᵀX)⁻¹ Xᵀ y。

        在特征矩阵前面加一列全 1 表示偏置，一次矩阵运算解出全部参数。
        theta[0] 是截距，theta[1:] 是各特征的系数。
        用 np.linalg.pinv（伪逆）代替普通逆，遇到奇异矩阵也稳定。
        """
        X = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        Xb = self._add_bias(X)                       # (n_samples, n_features+1)
        theta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y  # (XᵀX)⁻¹Xᵀy
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def fit_gradient_descent(self, X, y, lr=0.01, n_iters=1000):
        """批量梯度下降迭代求解。

        每一步对 MSE 损失求梯度，沿负梯度方向更新参数：
            梯度 = (2/n) · Xᵀ(X·theta - y)
            theta ← theta - lr · 梯度

        参数
        ----
        lr : float
            学习率，步长太大不收敛、太小收敛慢。
        n_iters : int
            迭代次数。
        """
        X = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        Xb = self._add_bias(X)
        n_samples = Xb.shape[0]

        theta = np.zeros(Xb.shape[1])
        for _ in range(n_iters):
            residual = Xb @ theta - y                # 预测误差
            grad = (2.0 / n_samples) * (Xb.T @ residual)
            theta = theta - lr * grad

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        """用训练好的参数预测：y = X·coef_ + intercept_。

        返回形状为 (n_samples,) 的一维预测向量。
        """
        if self.coef_ is None:
            raise ValueError("模型尚未训练，请先调用 fit / fit_normal_equation / fit_gradient_descent。")
        X = self._as_2d(X)
        return X @ self.coef_ + self.intercept_
