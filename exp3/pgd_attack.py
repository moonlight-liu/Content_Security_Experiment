import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os


# 1. 依然是加载我们的 LeNet 网络
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


# 2. PGD 核心算法实现
def pgd_attack(model, images, labels, eps, alpha, iters):
    # 记录原始图像，用于后面的“投影（限制范围）”
    ori_images = images.data

    # 复制一份图片用于迭代修改
    adv_images = images.clone().detach()

    # 迭代 iters 次
    for i in range(iters):
        adv_images.requires_grad = True

        # 前向传播
        outputs = model(adv_images)
        model.zero_grad()

        # 计算损失
        loss = F.nll_loss(outputs, labels)

        # 反向传播获取梯度
        loss.backward()

        # 核心逻辑：小步长 alpha 朝梯度方向前进
        adv_images = adv_images + alpha * adv_images.grad.sign()

        # 限制扰动范围不能超过 eps (这就是 PGD 中的 Projected 投影操作)
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        # 将扰动加回原图，并限制像素值在 [0, 1] 之间
        adv_images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return adv_images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_dir = os.path.dirname(__file__)

    # 加载已训练的模型
    model = LeNet().to(device)
    model.load_state_dict(
        torch.load(os.path.join(base_dir, "lenet_mnist_model.pth"), map_location=device)
    )
    model.eval()

    # 加载测试集
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root=os.path.join(base_dir, "MNIST"),
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 设置 PGD 的超参数
    epsilon = 0.3  # 最大总扰动范围 (和刚才 FGSM 同样的大小，方便对比)
    alpha = 0.01  # 每次走的小步长
    iters = 40  # 迭代次数

    correct = 0
    total = 0

    print(f"开始进行 PGD 攻击测试...")
    print(f"参数设置: Epsilon={epsilon}, Alpha={alpha}, 迭代次数={iters}")

    # 为了节省演示时间，我们只攻击测试集的前 1000 张图片
    # 如果想跑全集，把下面的限制去掉即可
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= 1000:
            break

        data, target = data.to(device), target.to(device)

        # 先看看原图模型能不能测对
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue  # 原本就猜错了，不计入对抗样本的功劳

        # 生成 PGD 对抗样本
        perturbed_data = pgd_attack(model, data, target, epsilon, alpha, iters)

        # 再次送入模型测试
        output_adv = model(perturbed_data)
        final_pred = output_adv.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
        total += 1

        if total % 100 == 0:
            print(f"已攻击 {total} 张图片...")

    acc = correct / total
    print(f"\nPGD 攻击结束！")
    print(f"在 {total} 张原始分类正确的图片中，抵御住攻击的有 {correct} 张。")
    print(f"PGD 攻击下的模型准确率为: {acc * 100:.2f}%")
    print(
        "对比提示：在 Epsilon=0.3 时，FGSM 的准确率大概是 3.9%，你可以看看 PGD 是不是更厉害（准确率更低）！"
    )


if __name__ == "__main__":
    main()
