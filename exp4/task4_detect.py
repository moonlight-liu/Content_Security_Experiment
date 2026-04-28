from transformers import pipeline
import os

# 消除下载网络报错
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("正在加载 GPT-2 Output Detector (RoBERTa微调版) 模型，需要一点时间下载...")
# 加载专门针对 GPT-2 生成文本的检测模型
detector = pipeline("text-classification", model="roberta-base-openai-detector")

# 准备测试样本：长短不一的 AI 文本 vs 人类真实文本
texts = {
    "【1】极短文本 (AI生成)": "Who is Liu Xiang? He is a young man who is very handsome.",
    "【2】长文本 (AI生成，来自任务1)": "Who is Liu Xiang? He is a famous Chinese ichthyologist, a professor of medicine at Harvard Medical School, and a member of the National Academy of Sciences. The question, as I have already shown, is precisely this: What is Liu Xiang's name? His name is Liu Xiang. He is one of the most important figures in Hong Kong.",
    "【3】长文本 (人类真实写的百科)": "Liu Xiang is a Chinese former 110 meter hurdler. Liu is an Olympic Gold medalist and World Champion. His 2004 Olympic gold medal was the first in a men's track and field event for China.",
}

print("\n====== 实验4：文本检测结果 ======")
for desc, text in texts.items():
    # 进行预测
    result = detector(text)[0]

    # 获取标签和置信度得分 (Fake 代表 AI 生成，Real 代表人类编写)
    label = result["label"]
    score = result["score"]

    print(f"\n测试样本: {desc}")
    print(f"检测判断: {label} (真实度/伪造度概率: {score:.4f})")
    print(f"文本内容: {text}")
