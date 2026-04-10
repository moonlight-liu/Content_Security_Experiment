# 内容安全实验二：音频处理与分析

本项目是内容安全课程第二次实验的代码实现，主要包括两部分：

1. 单音频的时域波形、线性声谱图和 MFCC 可视化。
2. 基于 UrbanSound8K 数据集的音频分类训练与评估。

## 1. 项目结构

- [task1_features.py](task1_features.py)：任务一，读取单段音频并绘制波形图、线性频谱图和 MFCC 图。
- [task2_preprocess.py](task2_preprocess.py)：任务二，遍历 UrbanSound8K 数据集，提取 MFCC 均值特征并保存为 numpy 文件。
- [task2_train.py](task2_train.py)：任务二，使用 PyTorch 构建 MLP 模型进行 10 分类训练，并绘制训练曲线。
- [UrbanSound8K/](UrbanSound8K/)：数据集目录，需包含 `audio/` 和 `metadata/` 子目录。
- [X_features.npy](X_features.npy) / [y_labels.npy](y_labels.npy)：预处理后生成的特征文件。

## 2. 环境依赖

建议使用 Python 3.8 及以上版本，并安装以下依赖：

```bash
pip install librosa torch matplotlib numpy pandas tqdm scikit-learn
```

## 3. 运行流程

### 3.1 单音频特征可视化

准备一段 `.wav` 音频，修改 [task1_features.py](task1_features.py) 中的音频路径，然后运行：

```bash
python task1_features.py
```

### 3.2 数据集特征提取

确认 UrbanSound8K 数据集位于当前目录下的 [UrbanSound8K/](UrbanSound8K/) 文件夹中，然后运行：

```bash
python task2_preprocess.py
```

运行结束后会生成：

- [X_features.npy](X_features.npy)
- [y_labels.npy](y_labels.npy)

### 3.3 模型训练与评估

运行训练脚本：

```bash
python task2_train.py
```

训练完成后会生成训练结果图 [task2_train_result.png](task2_train_result.png)，并在终端输出每轮训练进度、损失和测试准确率。

## 4. 实验结果

### 4.1 单音频特征图

单音频分析结果可见 [任务1结果图](任务1结果.png)。

### 4.2 训练结果

训练过程中的损失下降曲线和准确率上升曲线可见 [实验1结果图](实验1结果.png)。

从结果来看：

- 训练损失持续下降，说明模型逐步收敛。
- 测试集准确率稳定上升并达到较高水平，说明模型具有较好的分类能力。
- 后期曲线趋于平稳，说明继续增加训练轮数带来的收益有限。

## 5. 结论

- MFCC 能够有效提取音频中的关键频谱特征，并降低后续模型输入维度。
- 基于 MFCC 特征和 MLP 模型，可以较好地完成 UrbanSound8K 的城市环境声分类任务。
- 本实验表明，使用传统特征提取 + 轻量神经网络的方案，能够在较低计算成本下获得不错的分类效果。

## 6. 参考资料

- 课程课件《音频处理与分析》
- Librosa 官方文档
- PyTorch 官方文档