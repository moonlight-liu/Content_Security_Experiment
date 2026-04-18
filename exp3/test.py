import torch
import sys


def check_pytorch_gpu():
    print("===== PyTorch & GPU 环境检测 =====")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")

    # 1. 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 是否可用: {cuda_available}")

    if not cuda_available:
        print("\n❌ 警告：CUDA 不可用，PyTorch 只能使用 CPU！")
        print("可能原因：")
        print("1. 未安装 GPU 版 PyTorch")
        print("2. NVIDIA 驱动未安装或版本过低")
        print("3. CUDA  toolkit 未安装")
        return

    # 2. 检查 CUDA 版本
    cuda_version = torch.version.cuda
    print(f"CUDA 版本: {cuda_version}")

    # 3. GPU 设备数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU 数量: {gpu_count}")

    # 4. 遍历所有 GPU 信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # 转 GB
        print(f"\nGPU {i}: {gpu_name}")
        print(f"显存大小: {gpu_memory:.2f} GB")

    # 5. 测试张量能否在 GPU 上创建（最关键的验证）
    try:
        x = torch.tensor([1.0, 2.0]).cuda()
        print(f"\n✅ 测试成功！张量已在 GPU 上创建: {x}")
        print(f"✅ 当前使用的设备: {x.device}")
        print("\n🎉 全部正常！PyTorch + GPU 加速已配置完成！")
    except Exception as e:
        print(f"\n❌ GPU 测试失败，错误信息: {e}")


if __name__ == "__main__":
    check_pytorch_gpu()
