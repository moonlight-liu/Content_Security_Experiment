import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 解决matplotlib中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 使用脚本所在目录作为统一基准，避免在不同终端路径下运行时报错
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_FEATURES_PATH = os.path.join(BASE_DIR, "X_features.npy")
Y_LABELS_PATH = os.path.join(BASE_DIR, "y_labels.npy")
RESULT_PNG = os.path.join(BASE_DIR, "task2_train_result.png")

# 不使用 GPU，明确固定在 CPU 上运行
DEVICE = torch.device("cpu")

# ==========================================
# 1. 数据加载与预处理
# ==========================================
print("正在加载特征数据...")
X = np.load(X_FEATURES_PATH)
y = np.load(Y_LABELS_PATH)

# 将数据划分为训练集 (80%) 和测试集 (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 将Numpy数组转换为PyTorch的张量 (Tensor)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 使用DataLoader进行批次化包装，batch_size决定每次喂给模型多少条数据
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ==========================================
# 2. 定义深度学习模型 (多层感知机 MLP)
# ==========================================
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # 定义神经网络结构：输入层40 -> 隐藏层256 -> 隐藏层128 -> 输出层10(因为有10个类别)
        self.network = nn.Sequential(
            nn.Linear(40, 256),  # 第一层线性变换
            nn.ReLU(),  # 激活函数，增加非线性
            nn.Dropout(0.3),  # 随机丢弃30%神经元，防止过拟合
            nn.Linear(256, 128),  # 第二层线性变换
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(128, 10),  # 输出层，10个分类
        )

    def forward(self, x):
        # 前向传播过程
        return self.network(x)


# 实例化模型
model = AudioClassifier().to(DEVICE)

# ==========================================
# 3. 设置损失函数和优化器
# ==========================================
# 交叉熵损失函数，多分类任务的标配
criterion = nn.CrossEntropyLoss()
# Adam优化器，自动调节学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. 训练模型
# ==========================================
epochs = 50  # 训练轮数
train_losses = []
test_accuracies = []

print("\n开始训练模型...")
epoch_bar = tqdm(range(epochs), desc="训练进度", ncols=100)
for epoch in epoch_bar:
    model.train()  # 设置为训练模式
    running_loss = 0.0

    # 遍历训练数据
    batch_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, ncols=100
    )
    for inputs, labels in batch_bar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 前向传播得到预测值
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ==========================================
    # 5. 在测试集上评估模型
    # ==========================================
    model.eval()  # 设置为评估模式（关闭Dropout）
    correct = 0
    total = 0
    with torch.no_grad():  # 评估阶段不需要计算梯度
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            # 获取概率最大的那个类别的索引作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    epoch_bar.set_postfix(loss=f"{avg_train_loss:.4f}", acc=f"{accuracy:.2f}%")

    # 每10轮打印一次进度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%"
        )

print("\n模型训练结束！")

# ==========================================
# 6. 绘制训练结果图表
# ==========================================
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss", color="red")
plt.title("模型训练损失下降曲线")
plt.xlabel("Epoch (轮数)")
plt.ylabel("Loss (损失)")
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label="Test Accuracy", color="blue")
plt.title("模型测试集准确率上升曲线")
plt.xlabel("Epoch (轮数)")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.savefig(RESULT_PNG, dpi=200)
plt.show()
print(f"训练结果图已保存到: {RESULT_PNG}")
