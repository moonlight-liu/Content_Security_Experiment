import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os


# 1. 重新定义网络结构
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


# 用于将 PyTorch 的 Tensor 转换为 Matplotlib 可以画出的图片格式
def show_tensor_images(tensor_img, title, filename):
    # 将 tensor 转换到 CPU 并转为 numpy 数组
    npimg = tensor_img.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    # PyTorch的图像格式是 (C, H, W)，Matplotlib需要 (H, W, C)，所以要转置一下
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.title(title, fontsize=16)
    plt.axis("off")  # 不显示坐标轴
    plt.savefig(filename, bbox_inches="tight")
    print(f"已保存图像: {filename}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(__file__)

    # 加载模型
    model = LeNet().to(device)
    pretrained_model = os.path.join(base_dir, "lenet_mnist_model.pth")
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    model.eval()

    # 加载测试集，这里设 batch_size=64 刚好可以画一个 8x8 的格子（和PPT第6页一样）
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root=os.path.join(base_dir, "MNIST"),
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 获取一个批次的数据
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    # 允许计算输入数据的梯度
    data.requires_grad = True

    # 前向传播并计算损失
    output = model(data)
    loss = F.nll_loss(output, target)

    # 反向传播获取梯度
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data

    # 1. 提取梯度符号 (这就是攻击噪声)
    sign_data_grad = data_grad.sign()

    # 2. 生成对抗样本 (这里我们选 epsilon=0.3 作为演示，此时攻击成功率很高，且肉眼能看出噪点)
    epsilon = 0.3
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # 限制像素范围在 0-1 之间

    # --- 开始画图 ---

    # (1) 原始数据梯度符号 (需要把 -1 到 1 的范围映射到 0 到 1 以便显示)
    normalized_sign_grad = (sign_data_grad + 1) / 2
    grid_sign_grad = make_grid(normalized_sign_grad, nrow=8, padding=2, normalize=False)
    show_tensor_images(
        grid_sign_grad,
        "Gradient Sign (Noise)",
        os.path.join(base_dir, "gradient_sign.png"),
    )

    # (2) 添加扰动后的对抗样本
    grid_perturbed = make_grid(perturbed_data, nrow=8, padding=2, normalize=False)
    show_tensor_images(
        grid_perturbed,
        f"Adversarial Examples (epsilon={epsilon})",
        os.path.join(base_dir, "adversarial_examples.png"),
    )


if __name__ == "__main__":
    main()
