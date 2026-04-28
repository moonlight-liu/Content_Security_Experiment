import os

# 消除下载网络报错
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM

print("正在加载模型和分词器...")
# 加载 GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 作为提示词
prompt = "Who is Liu Xiang? He is"
# 将文本转换为模型认识的数字 ID (Token)
inputs = tokenizer(prompt, return_tensors="pt")

# 设置生成的基础参数
gen_kwargs = {
    "max_new_tokens": 40,  # 每次往后生成40个词
    "pad_token_id": tokenizer.eos_token_id,  # 消除警告
    "do_sample": True,  # 【关键】必须开启采样模式，下面调的参数才起作用！
}

print("\n====== 实验2：参数调整测试 ======\n")

# 【1】标准随机采样
print("【1】标准生成 (Temperature=1.0, Top-K=50)")
out1 = model.generate(**inputs, **gen_kwargs, temperature=1.0, top_k=50, top_p=1.0)
print(tokenizer.decode(out1[0], skip_special_tokens=True) + "\n")

# 【2】低 Temperature (接近0)
print(
    "【2】低 Temperature (Temperature=0.1) -> 预期：文本非常肯定，可能开始重复、像百科全书"
)
out2 = model.generate(**inputs, **gen_kwargs, temperature=0.1, top_k=50, top_p=1.0)
print(tokenizer.decode(out2[0], skip_special_tokens=True) + "\n")

# 【3】高 Temperature
print("【3】高 Temperature (Temperature=2.0) -> 预期：脑洞大开，胡言乱语，拼写错误")
out3 = model.generate(**inputs, **gen_kwargs, temperature=2.0, top_k=50, top_p=1.0)
print(tokenizer.decode(out3[0], skip_special_tokens=True) + "\n")

# 【4】Top-K 限制
print(
    "【4】严格 Top-K (Top-K=3) -> 预期：每次只在最可能的3个词里选，限制了天马行空，比较通顺"
)
out4 = model.generate(**inputs, **gen_kwargs, temperature=1.0, top_k=3, top_p=1.0)
print(tokenizer.decode(out4[0], skip_special_tokens=True) + "\n")

# 【5】Top-P 限制
# 这里把 top_k 设为 0 是为了排除干扰，单独看 top_p 的效果
print(
    "【5】Top-P 限制 (Top-P=0.5) -> 预期：动态截断，累积概率达到0.5就不往后看了，平衡了通顺与多样性"
)
out5 = model.generate(**inputs, **gen_kwargs, temperature=1.0, top_k=0, top_p=0.5)
print(tokenizer.decode(out5[0], skip_special_tokens=True) + "\n")
