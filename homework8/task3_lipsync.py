"""
任务3：运行 Wav2Lip 进行唇形同步
将特朗普视频与中文语音进行唇形同步
"""

import subprocess
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")

# 文件路径
VIDEO_PATH = os.path.join(BASE_DIR, "gettyimages-2072456276-640_adpp.mp4")
AUDIO_PATH = os.path.join(BASE_DIR, "speech.wav")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "wav2lip_gan.pt")  # 注意后缀是 .pt
OUTPUT_PATH = os.path.join(BASE_DIR, "lipsync_out.avi")


def run_wav2lip():
    """运行 Wav2Lip 推理"""
    # 检查所有文件是否存在
    for name, path in [
        ("视频", VIDEO_PATH),
        ("音频", AUDIO_PATH),
        ("模型", CHECKPOINT_PATH),
    ]:
        if not os.path.exists(path):
            print(f"错误: {name}文件不存在: {path}")
            sys.exit(1)

    if not os.path.exists(WAV2LIP_DIR):
        print(f"错误: Wav2Lip目录不存在: {WAV2LIP_DIR}")
        sys.exit(1)

    print("=" * 50)
    print("Wav2Lip 唇形同步推理")
    print("=" * 50)
    print(f"视频: {VIDEO_PATH}")
    print(f"音频: {AUDIO_PATH}")
    print(f"模型: {CHECKPOINT_PATH}")
    print(f"输出: {OUTPUT_PATH}")
    print(f"设备: CUDA (RTX 4060)")
    print("=" * 50)

    # 构建命令
    cmd = [
        sys.executable, "inference.py",
        "--checkpoint_path", CHECKPOINT_PATH,
        "--face", VIDEO_PATH,
        "--audio", AUDIO_PATH,
        "--outfile", OUTPUT_PATH,
        "--pads", "0", "20", "0", "0",    # 底部加padding确保下巴完整
        "--resize_factor", "2",            # 缩小2倍处理（人脸较小）
    ]

    print("正在运行 Wav2Lip，预计需要 3-8 分钟...\n")

    # 切换到 Wav2Lip 目录执行
    original_dir = os.getcwd()
    os.chdir(WAV2LIP_DIR)

    try:
        result = subprocess.run(cmd, check=True)
        print("\nWav2Lip 处理完成！")
    except subprocess.CalledProcessError as e:
        print(f"\nWav2Lip 运行失败，返回码: {e.returncode}")
        sys.exit(1)
    finally:
        os.chdir(original_dir)

    # 检查输出
    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"输出文件大小: {size_mb:.1f} MB")
    else:
        print(f"警告: 未找到输出文件 {OUTPUT_PATH}")


if __name__ == "__main__":
    run_wav2lip()
