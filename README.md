# ml-from-scratch

从零开始学机器学习 —— 用 NumPy 亲手实现经典机器学习算法，配套《从零开始学机器学习》系列文章。

## 这个仓库是干什么的

不追求性能，也不追求覆盖所有算法，追求一件事：**把每个经典算法的核心逻辑用最朴素的 NumPy 亲手写一遍**，让你看到的每一行都不是黑盒。写完再用 scikit-learn 跑同样的东西做对照——你手写的原理，工业界一行代码就调用。

每个算法模块都自包含：核心实现 + 最小可跑示例 + 单元测试 + 导出真实运行数据的绘图脚本（文章里的每张图都来自代码真实输出，不是画的）。

## 目录结构

```
ml-from-scratch/
├── common/           # 公共工具：数据集生成、绘图字体辅助
├── foundations/      # 地基：损失与拟合、梯度下降、过拟合与正则化
├── supervised/       # 监督学习：线性/逻辑回归、KNN、朴素贝叶斯、决策树、SVM
├── ensemble/         # 集成学习：随机森林、GBDT、模型评估
├── unsupervised/     # 无监督：K-Means、PCA、异常检测
├── neural_net/       # 通向神经网络：感知机、反向传播、手写 MLP 识别 MNIST
└── docs/
    ├── references.md # 经典书籍锚点 + 开源项目清单
    └── articles/     # 配套系列文章 markdown（与公众号双发）
```

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest -v                      # 跑全部测试
python -m foundations.export_charts      # 导出地基篇的真实数据配图
```

## 参考资料

代码根据经典教材和公开课的原理从零重新实现，**只借鉴思路和边界处理，不摘抄任何开源项目源码**。参考清单见 [docs/references.md](docs/references.md)，主要包括：周志华《机器学习》、李航《统计学习方法》、吴恩达 CS229，以及 scikit-learn（工业对照）、eriklindernoren/ML-From-Scratch、karpathy/micrograd（思路参考）。

## 开发约定

- **代码先行 + 真实数据配图：** 每个模块先实现、跑通、导出真实中间数据，这些数据直接作为配套文章的配图，可追溯到代码输出。
- **参考不摘抄：** 参考思路和边界，代码从原理重新实现。用 scikit-learn 处理同一输入做数值对照验证正确性。
- **测试驱动：** 每个模块先写测试，再写实现。

## License

MIT License，见 [LICENSE](LICENSE)。
