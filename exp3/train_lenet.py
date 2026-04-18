import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os


# 1. 搭建LeNet网络结构 (对应PPT第5页)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 输入 1通道(灰度图), 输出 10个特征图, 卷积核5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 输入 10通道, 输出 20个特征图, 卷积核5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # 全连接层
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


def train_model():
    # 设置超参数
    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载MNIST数据集
    # 统一将数据存放在 exp3/MNIST 目录下
    data_root = os.path.join(os.path.dirname(__file__), "MNIST")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 3. 初始化模型和优化器
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    # 4. 训练循环
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    # 5. 保存训练好的模型权重
    torch.save(model.state_dict(), "lenet_mnist_model.pth")
    print("模型训练完成，已保存为 lenet_mnist_model.pth")


if __name__ == "__main__":
    train_model()
