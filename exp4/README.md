# GPT-2 文本生成与检测实验

本仓库包含信息内容安全实验课的相关代码实现，主要涵盖 GPT-2 模型的文本生成、参数调优、注意力机制分析以及 AI 生成内容检测。

## 📁 目录结构

*   `task1_generate.py`: **任务 1 - 基础文本生成**。使用 GPT-2 默认参数生成一段关于“刘翔”的文本。
*   `task2_params.py`: **任务 2 - 生成参数对比实验**。通过控制变量法，对比 Temperature、Top-K、Top-P 等参数对生成效果的影响。
*   `task4_detect.py`: **任务 4 - AI 文本检测**。使用基于 RoBERTa 微调的检测器对 AI 生成文本和人类文本进行真伪判定。
*   `README.md`: 项目说明文档。

## 🛠 环境准备

本项目基于 Python 3.8+ 环境开发，建议使用 `conda` 或 `venv` 管理虚拟环境。

### 1. 安装依赖库
在终端运行以下命令安装必要的深度学习与模型库：
```bash
pip install transformers torch
```

### 2. 网络配置（重要）
由于 HuggingFace 官方服务器在国内访问受限，代码中已包含以下配置以使用国内镜像源：
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

## 🚀 运行说明

### 任务 1：基础生成
运行脚本查看 GPT-2 的初始生成能力：
```bash
python task1_generate.py
```

### 任务 2：参数调优
观察不同参数下的生成差异（如低温度导致的“复读机”现象）：
```bash
python task2_params.py
```

### 任务 4：文本检测
测试检测器对不同长度、不同来源文本的判别置信度：
```bash
python task4_detect.py
```

## 📝 实验观察要点

1.  **任务 1**：观察模型生成的“幻觉”现象（如职业误报）。
2.  **任务 2**：重点对比 `Temperature=0.1`（确定性）与 `Temperature=2.0`（发散性）的文本差异。
3.  **任务 4**：分析长短文本对检测分数的影响，观察对正式百科体文本的误判现象。

## 📚 参考资料
*   [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/index)
*   [OpenAI GPT-2 原始仓库](https://github.com/openai/gpt-2)
*   [HF-Mirror 镜像站](https://hf-mirror.com/)
