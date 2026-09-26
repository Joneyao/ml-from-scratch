"""反向传播——micrograd 风格的极简标量自动微分引擎。

反向传播的本质是**链式法则 + 计算图**。

每做一次运算（加、乘、幂、激活），我们不只算出结果，
还把这次运算和它的输入连成一张有向图：结果节点指向它的父节点。
整张图记录了"输出是怎么由输入一步步算出来的"。

求梯度时，我们从输出节点出发，把它对自己的梯度设为 1，
然后沿着图**反向**流动：每条边上乘以这一步运算的 local gradient
（局部导数），一路把梯度累加到每个节点的 grad 上。
这正是链式法则——复合函数的导数是各层导数的连乘。

例如 z = x * y：
    ∂z/∂x = y，∂z/∂y = x
反向传播时，x.grad += y * z.grad，y.grad += x * z.grad。
"""


class Value:
    """包装一个标量，参与自动微分。

    记录：
      - data：标量的前向数值
      - grad：损失对该节点的梯度（反向传播后填充，初始为 0）
      - _prev：构建它的父节点集合（计算图的边）
      - _op：产生它的运算名（便于调试/可视化）
      - _backward：一个闭包，把本节点的 grad 按 local gradient
                   反向分配给父节点
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        # 叶子节点没有父节点，反向时什么都不做
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """加法：out = self + other。

        local gradient 都是 1，梯度原样流回两个加数。
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """乘法：out = self * other。

        ∂out/∂self = other，∂out/∂other = self，即交叉相乘。
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        """幂运算：out = self ** exponent（exponent 为常数）。

        ∂out/∂self = exponent * self**(exponent-1)。
        """
        assert isinstance(exponent, (int, float)), "指数只支持 int/float 常数"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """双曲正切激活：out = tanh(self)。

        导数 ∂out/∂self = 1 - tanh(self)**2 = 1 - out**2。
        """
        import math

        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """ReLU 激活：out = max(0, self)。

        导数：self>0 时为 1，否则为 0。
        """
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ---- 一些便利运算，方便写表达式 ----
    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def backward(self):
        """从本节点出发反向传播，填充计算图中所有节点的 grad。

        步骤：
          1. 对计算图做拓扑排序，保证在处理某节点前，
             所有依赖它的下游节点都已处理完（梯度已就绪）。
          2. 把输出节点对自己的梯度设为 1（∂out/∂out = 1）。
          3. 按拓扑逆序依次调用每个节点的 _backward，
             沿链式法则把梯度分配给父节点。
        """
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
