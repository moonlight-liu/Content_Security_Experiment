"""
任务4：压缩视频至10MB以下
使用 FFmpeg 进行多轮压缩，确保最终文件不超过10MB
"""

import subprocess
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "lipsync_out.avi")
OUTPUT_PATH = os.path.join(BASE_DIR, "final_result.mp4")

MAX_SIZE = 10 * 1024 * 1024  # 10MB


def compress(crf, resolution, fps, audio_bitrate):
    """使用指定参数压缩视频"""
    output = os.path.join(BASE_DIR, f"temp_{crf}_{resolution}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", INPUT_PATH,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-vf", f"scale={resolution}",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", f"{audio_bitrate}k",
        "-ac", "1",
        output
    ]

    print(f"  压缩参数: CRF={crf}, 分辨率={resolution}, {fps}fps, 音频={audio_bitrate}k")
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL)

    size = os.path.getsize(output)
    print(f"  输出大小: {size / (1024*1024):.2f} MB")
    return output, size


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"错误: 未找到输入文件 {INPUT_PATH}")
        print("请先运行 task3_lipsync.py 生成唇形同步视频")
        sys.exit(1)

    print("=" * 50)
    print("视频压缩 - 目标: ≤10MB")
    print("=" * 50)
    print(f"输入: {INPUT_PATH}")
    print(f"输入大小: {os.path.getsize(INPUT_PATH) / (1024*1024):.1f} MB")
    print("=" * 50)

    # 压缩策略列表（从优到劣）
    strategies = [
        # (CRF, 分辨率, fps, 音频码率)
        (26, "854:480", 25, 64),
        (28, "854:480", 25, 64),
        (28, "640:360", 25, 48),
        (30, "640:360", 25, 48),
        (30, "640:360", 20, 32),
    ]

    last_output = None
    for crf, res, fps, abit in strategies:
        print(f"\n尝试压缩方案...")
        output, size = compress(crf, res, fps, abit)

        if size <= MAX_SIZE:
            # 成功！替换最终输出
            if os.path.exists(OUTPUT_PATH):
                os.remove(OUTPUT_PATH)
            os.rename(output, OUTPUT_PATH)
            last_output = OUTPUT_PATH
            print(f"\n✅ 压缩成功！最终输出: {OUTPUT_PATH}")
            print(f"   文件大小: {size / (1024*1024):.2f} MB / 10 MB")
            break
        else:
            # 删除临时文件，继续下一轮
            os.remove(output)
            print(f"   ❌ 超过10MB，尝试更激进的压缩...")
    else:
        # 所有策略都失败，用最激进的参数
        print("\n使用极限压缩参数...")
        cmd = [
            "ffmpeg", "-y",
            "-i", INPUT_PATH,
            "-c:v", "libx264",
            "-preset", "veryslow",
            "-crf", "32",
            "-vf", "scale=480:272",
            "-r", "20",
            "-c:a", "aac",
            "-b:a", "24k",
            "-ac", "1",
            OUTPUT_PATH
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL)
        final_size = os.path.getsize(OUTPUT_PATH)
        print(f"\n极限压缩完成: {final_size / (1024*1024):.2f} MB")

    # 最终验证
    if os.path.exists(OUTPUT_PATH):
        size = os.path.getsize(OUTPUT_PATH)
        if size <= MAX_SIZE:
            print(f"✅ 最终文件 {OUTPUT_PATH}")
            print(f"   大小: {size / (1024*1024):.2f} MB (满足 ≤10MB 要求)")
        else:
            print(f"⚠️  文件仍超过10MB: {size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
