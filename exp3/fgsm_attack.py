import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# 1. 重新定义我们刚才训练的网络结构，以便加载权重
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 2. 定义FGSM攻击函数 (核心！！对应PPT第5页的公式)
def fgsm_attack(image, epsilon, data_grad):
    # 获取数据梯度的符号 (1 或 -1)
    sign_data_grad = data_grad.sign()
    # 原始图像 加上 扰动 (epsilon * 符号)
    perturbed_image = image + epsilon * sign_data_grad
    # 将像素值强行限制在 [0, 1] 的范围内，因为图像像素不能越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# 3. 定义测试循环，用于在不同 epsilon 下测试准确率
def test(model, device, test_loader, epsilon):
    correct = 0  # 记录分类正确的数量

    # 遍历测试集里的每一张图片
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # 这一步非常关键：告诉PyTorch我们需要计算并保留输入图片的梯度
        data.requires_grad = True

        # 将图片输入给模型，得到初始预测
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # 获取最大概率的类别索引

        # 如果原本模型就预测错了，那对抗攻击没有意义，直接跳过
        if init_pred[0].item() != target.item():
            continue

        # 计算预测和真实标签的损失
        loss = F.nll_loss(output, target)

        # 清空模型之前的梯度
        model.zero_grad()
        # 反向传播，但这次我们要的是对输入 data 的梯度！
        loss.backward()

        # 提取图片的梯度
        data_grad = data.grad.data

        # 调用我们写的 FGSM 函数生成对抗样本（加了噪点的图）
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 把生成的对抗样本再次输入给模型
        output = model(perturbed_data)

        # 查看模型这次预测的结果
        final_pred = output.max(1, keepdim=True)[1]

        # 如果预测结果仍然是正确的标签，说明模型抗住了攻击
        if final_pred.item() == target.item():
            correct += 1

    # 计算当前 epsilon 下的整体准确率
    final_acc = correct / float(len(test_loader))
    print(
        f"Epsilon: {epsilon}\t测试集准确率(Accuracy) = {correct} / {len(test_loader)} = {final_acc * 100:.2f}%"
    )
    return final_acc


def main():
    # 按照 PPT 第 7 页，设置一系列的扰动大小
    epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    base_dir = os.path.dirname(__file__)
    pretrained_model = os.path.join(base_dir, "lenet_mnist_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并加载权重
    model = LeNet().to(device)
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    # 将模型设置为评估模式 (很重要，会关闭Dropout，保证测试结果稳定)
    model.eval()

    # 加载测试集，注意这里 batch_size=1，必须一张一张攻击
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root=os.path.join(base_dir, "MNIST"),
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    accuracies = []

    # 遍历所有的 epsilon，对模型进行轮番攻击
    print("开始进行 FGSM 对抗攻击测试...")
    for eps in epsilons:
        acc = test(model, device, test_loader, eps)
        accuracies.append(acc)

    # 4. 绘制并保存 PPT 第 7 页要求的 折线图
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, accuracies, "o-")  # o- 表示带圆点的实线
    plt.title("Accuracy vs Epsilon (FGSM Attack)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "fgsm_accuracy_curve.png"))
    print("\n折线图已生成并保存为：fgsm_accuracy_curve.png")


if __name__ == "__main__":
    main()
