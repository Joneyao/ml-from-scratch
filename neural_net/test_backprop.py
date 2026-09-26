"""反向传播引擎的测试：前向计算、手推梯度、数值梯度校验。"""
import math

from neural_net.backprop import Value


def test_forward_multiply():
    # 前向：两个标量相乘，data 应为 6.0
    out = Value(2.0) * Value(3.0)
    assert out.data == 6.0


def test_backward_matches_manual_derivative():
    # z = x*y + x，x=2, y=3
    # ∂z/∂x = y + 1 = 4，∂z/∂y = x = 2
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()
    assert math.isclose(x.grad, 4.0, rel_tol=1e-9)
    assert math.isclose(y.grad, 2.0, rel_tol=1e-9)


def _numerical_grad(f, x0, eps=1e-6):
    """中心差分估计 f 在 x0 处的数值梯度。"""
    return (f(x0 + eps) - f(x0 - eps)) / (2 * eps)


def test_backward_matches_numerical_gradient():
    # 一个带激活函数的两层小表达式：
    #   h = tanh(a*w1 + b1)
    #   out = relu(h*w2 + b2)
    # 校验 out 对 a 的解析梯度与有限差分接近。
    a_val, w1_val, b1_val, w2_val, b2_val = 0.5, -1.2, 0.3, 0.8, -0.1

    def build(a_scalar):
        a = Value(a_scalar)
        w1 = Value(w1_val)
        b1 = Value(b1_val)
        w2 = Value(w2_val)
        b2 = Value(b2_val)
        h = (a * w1 + b1).tanh()
        out = (h * w2 + b2).relu()
        return a, out

    a, out = build(a_val)
    out.backward()
    analytic = a.grad

    numeric = _numerical_grad(lambda av: build(av)[1].data, a_val)
    assert abs(analytic - numeric) < 1e-4
