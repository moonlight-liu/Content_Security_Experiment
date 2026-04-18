import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os


# 1. 模型定义（保持不变）
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


# 2. FGSM 攻击函数（用作训练过程中的“疫苗生成器”）
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def adv_train_model():
    batch_size = 64
    epochs = 20
    learning_rate = 0.01
    epsilon = 0.3  # 训练时使用的对抗扰动强度

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_dir = os.path.dirname(__file__)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=os.path.join(base_dir, "MNIST"),
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 初始化一个新的模型
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    print("开始进行对抗训练 (Adversarial Training)...")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # --- 阶段一：动态生成对抗样本 (打疫苗) ---
            data.requires_grad = True
            model.eval()  # 生成对抗样本时需要固定网络状态
            output_for_attack = model(data)
            loss_for_attack = F.nll_loss(output_for_attack, target)
            model.zero_grad()
            loss_for_attack.backward()
            data_grad = data.grad.data

            # 生成带噪点的图片
            perturbed_data = fgsm_attack(data, epsilon, data_grad).detach()

            # --- 阶段二：混合训练 (一半干净数据，一半对抗数据) ---
            model.train()
            optimizer.zero_grad()

            # 1. 计算干净样本的损失
            output_clean = model(data)
            loss_clean = F.nll_loss(output_clean, target)

            # 2. 计算对抗样本的损失
            output_adv = model(perturbed_data)
            loss_adv = F.nll_loss(output_adv, target)

            # 3. 总损失 = α * 干净损失 + (1-α) * 对抗损失 (论文中通常取 α=0.5)
            loss = 0.5 * loss_clean + 0.5 * loss_adv

            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(
                    f"Adv Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tTotal Loss: {loss.item():.6f}"
                )

    # 保存打过疫苗的“鲁棒模型”
    robust_path = os.path.join(base_dir, "lenet_mnist_robust.pth")
    torch.save(model.state_dict(), robust_path)
    print(f"\n对抗训练完成！具有防御能力的新模型已保存为 {robust_path}")


if __name__ == "__main__":
    adv_train_model()
