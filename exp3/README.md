# 图像对抗样本生成与检测 (Adversarial Examples on MNIST)

## 📌 实验简介
本项目基于 PyTorch 框架，在 MNIST 数据集上搭建了 LeNet 卷积神经网络，并完成了以下三个主要任务：
1. **基础攻击**：复现了经典的 **FGSM（快速梯度符号法）** 白盒攻击算法，并可视化了对抗扰动及对抗样本。
2. **进阶攻击**：实现并测试了更强力的 **PGD（投影梯度下降）** 迭代攻击算法。
3. **防御机制**：复现了基于 FGSM 的**对抗训练（Adversarial Training）**，成功提升了模型面对对抗攻击时的鲁棒性，同时起到了正则化效果，提升了模型在原始干净数据上的准确率。

---

## 🛠️ 环境依赖
在运行本代码前，请确保 Python 环境中已安装以下库：
* `torch` (支持 CUDA 更佳)
* `torchvision`
* `matplotlib`
* `numpy`

---

## 📂 目录与文件说明
确保当前目录结构如下（特别是 MNIST 数据集的存放路径）：

```text
exp3/
 │
 ├── MNIST/
 │    └── raw/
 │         ├── train-images-idx3-ubyte
 │         ├── train-labels-idx1-ubyte
 │         ├── t10k-images-idx3-ubyte
 │         └── t10k-labels-idx1-ubyte
 │
 ├── train_lenet.py        # 训练基础 LeNet 正常模型
 ├── fgsm_attack.py        # 执行 FGSM 攻击并绘制不同 epsilon 下的准确率折线图
 ├── fgsm_visualize.py     # 生成并保存梯度符号图与对抗样本图
 ├── pgd_attack.py         # 执行 PGD 攻击代码
 ├── adv_train_lenet.py    # 使用对抗训练策略重新训练具有防御力的模型
 └── test_defense.py       # 对比正常模型与防御模型在干净数据和攻击数据上的表现
```

---

## 🚀 运行步骤 (请严格按照顺序执行)

**第一步：训练基础模型**
```bash
python train_lenet.py
```
*功能：训练 5 轮 LeNet 模型，并在当前目录下生成基础模型权重 `lenet_mnist_model.pth`。*

**第二步：FGSM 攻击与可视化 (任务1)**
```bash
python fgsm_attack.py
python fgsm_visualize.py
```
*功能：验证 FGSM 攻击效果，生成准确率折线图 `fgsm_accuracy_curve.png`，以及对抗样本可视化图片 `gradient_sign.png` 和 `adversarial_examples.png`。*

**第三步：PGD 攻击测试 (任务2)**
```bash
python pgd_attack.py
```
*功能：在 epsilon=0.3 的条件下测试 PGD 迭代攻击，观察准确率的大幅下降（对比 FGSM 攻击能力更强）。*

**第四步：对抗训练 (任务3防御)**
```bash
python adv_train_lenet.py
```
*功能：将生成的 FGSM 对抗样本混入训练集，进行 20 轮对抗训练，生成具备防御力的模型权重 `lenet_mnist_robust.pth`。*

**第五步：验证防御效果**
```bash
python test_defense.py
```
*功能：在干净的测试集和施加 FGSM 攻击的测试集上，横向对比“基础模型”与“防御模型”的准确率。*

---

## 📊 核心实验结论
1. **攻击易行性**：极小的梯度扰动（肉眼难以察觉或表现为轻微噪点）即可导致原本准确率高达 96%+ 的深度学习模型准确率跌至 1% 以下。
2. **多步优于单步**：在相同的扰动预算 ($\epsilon=0.3$) 下，迭代式的 PGD 攻击比单步式的 FGSM 攻击具有更强的破坏力。
3. **防御有效性**：对抗训练能极大地提升模型鲁棒性（在攻击下准确率从 4.04% 回升至 79.26%）。同时，对抗训练起到了数据增强/正则化的作用，使得模型在原始干净数据集上的准确率不降反升（96.58% -> 97.37%）。

