import os

# 【重要提示】如果你在运行代码时遇到网络连接错误（huggingface.co无法访问），
# 请将下面这一行的注释符号（#）去掉，使用国内镜像源下载模型：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import pipeline

print("正在加载GPT-2模型，如果是第一次运行，需要几分钟时间下载模型文件...")
# 1. 创建文本生成管线，指定使用 "gpt2" 模型
generator = pipeline("text-generation", model="gpt2")

# 2. 设置一个开头（提示词），你可以把引号里的内容换成任何你感兴趣的英文句子
prompt = "Who is Liu Xiang? He is a famous Chinese "

print("模型加载完成！正在生成文本...")
# 3. 让模型接着你的开头续写
results = generator(
    prompt,
    max_length=50,  # 限制生成的总字符长度为50
    num_return_sequences=1,  # 只生成1个结果
    pad_token_id=50256,  # 消除由于pad_token没设置导致的控制台警告
)

# 4. 打印生成的结果
print("\n====== 生成结果 ======")
print(results[0]["generated_text"])
print("======================")
