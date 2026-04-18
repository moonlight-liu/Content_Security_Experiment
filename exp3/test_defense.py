import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os


# 1. 模型定义
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


# 2. FGSM 攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# 3. 评测函数
def evaluate_model(model, device, test_loader, epsilon, is_attack=False):
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        if is_attack:
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                continue

            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            # 受到攻击的图片
            test_data = fgsm_attack(data, epsilon, data_grad)
        else:
            # 干净的图片
            test_data = data

        output = model(test_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
        total += 1

    acc = correct / float(total) if total > 0 else 0
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_dir = os.path.dirname(__file__)

    # 加载两个模型
    normal_model = LeNet().to(device)
    normal_model.load_state_dict(
        torch.load(os.path.join(base_dir, "lenet_mnist_model.pth"), map_location=device)
    )
    normal_model.eval()

    robust_model = LeNet().to(device)
    robust_model.load_state_dict(
        torch.load(
            os.path.join(base_dir, "lenet_mnist_robust.pth"), map_location=device
        )
    )
    robust_model.eval()

    # 测试集 (为了评测准确，我们使用全量10000张测试)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root=os.path.join(base_dir, "MNIST"),
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    epsilon = 0.3

    print("\n" + "=" * 50)
    print("【第一回合：正常干净数据集上的表现 (对应任务3要求)】")
    acc_norm_clean = evaluate_model(
        normal_model, device, test_loader, 0, is_attack=False
    )
    print(f"-> 普通模型识别正常数字的准确率: {acc_norm_clean * 100:.2f}%")

    acc_robust_clean = evaluate_model(
        robust_model, device, test_loader, 0, is_attack=False
    )
    print(f"-> 防御模型识别正常数字的准确率: {acc_robust_clean * 100:.2f}%")
    print(
        "(注：防御模型在正常数据上的表现可能会有极其微小的下降，这是获得防御力付出的代价，属于正常现象)"
    )

    print("\n" + "=" * 50)
    print(f"【第二回合：FGSM 攻击下 (epsilon={epsilon}) 的表现】")
    acc_norm_adv = evaluate_model(
        normal_model, device, test_loader, epsilon, is_attack=True
    )
    print(f"-> 普通模型在攻击下的准确率: {acc_norm_adv * 100:.2f}%  <-- (惨不忍睹)")

    acc_robust_adv = evaluate_model(
        robust_model, device, test_loader, epsilon, is_attack=True
    )
    print(f"-> 防御模型在攻击下的准确率: {acc_robust_adv * 100:.2f}%  <-- (见证奇迹！)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
