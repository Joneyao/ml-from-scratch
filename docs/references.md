# 参考资料

本仓库的代码根据经典教材和公开课的原理**从零重新实现**，只借鉴思路和边界处理，不摘抄任何开源项目源码。

## 经典书籍（原理讲解的深度锚点）

| 书 / 课 | 一句话定位 | 主要参考章节 |
|---------|-----------|-------------|
| 周志华《机器学习》（西瓜书） | 中文读者最熟，覆盖全，通俗和公式兼顾 | 地基篇：第 1-2 章（模型评估与选择）；后续各算法对应各章 |
| 李航《统计学习方法》 | 公式推导严谨，研究生友好 | 第 1 章（统计学习方法概论）、各算法对应章 |
| 吴恩达 CS229 / Coursera ML | 直觉讲解的标杆 | 地基篇：Lecture 1-3（线性回归、梯度下降、正则化） |
| Hastie《The Elements of Statistical Learning》(ESL) | 深度参考，偏统计视角 | 按需查阅 |

## 开源教学项目（代码实现的对照 / 灵感）

**原则：只借鉴思路和边界处理，代码从原理重新实现，不复制源码。** 用 scikit-learn 处理同一输入做数值对照，验证正确性。

| 项目 | 用途 |
|------|------|
| [scikit-learn](https://scikit-learn.org/) | 工业界对照标准——每个算法手写后，用它跑同样的东西验证 |
| [eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) | 经典纯 NumPy 手写 ML 仓库，手写风格参考 |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | 神经网络篇（反向传播、MLP）的极简自动微分风格标杆 |
| [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms) | 手写实现思路参考 |
| [trekhleb/homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning) | 带讲解的手写实现参考 |

## 数据集

| 数据集 | 来源 | 用在哪 |
|--------|------|--------|
| iris / digits / breast_cancer | scikit-learn 内置 | 大部分监督/无监督篇 |
| 教学合成数据 | `common/datasets.py`（带噪声直线、三次多项式） | 地基篇 |
| MNIST | openml 或 torchvision | 神经网络篇（手写 MLP 识别手写数字） |
