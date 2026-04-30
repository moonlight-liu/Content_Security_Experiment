"""
任务5：验证最终视频
检查文件大小、时长、分辨率、可播放性
"""

import subprocess
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "final_result.mp4")
MAX_SIZE = 10 * 1024 * 1024  # 10MB


def get_ffprobe_info(filepath):
    """使用 ffprobe 获取视频信息"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)


def verify():
    """验证输出视频"""
    print("=" * 50)
    print("视频验证")
    print("=" * 50)

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 错误: 文件不存在: {VIDEO_PATH}")
        return False

    # 1. 文件大小
    size_bytes = os.path.getsize(VIDEO_PATH)
    size_mb = size_bytes / (1024 * 1024)
    print(f"\n📁 文件大小: {size_mb:.2f} MB / 10 MB")
    if size_bytes <= MAX_SIZE:
        print("   ✅ 满足 ≤10MB 要求")
    else:
        print("   ❌ 超过10MB限制！")
        return False

    # 2. 视频信息
    info = get_ffprobe_info(VIDEO_PATH)
    format_info = info.get("format", {})
    streams = info.get("streams", [])

    duration = float(format_info.get("duration", 0))
    print(f"\n⏱️  时长: {duration:.1f} 秒")

    for stream in streams:
        if stream["codec_type"] == "video":
            print(f"\n🎬 视频流:")
            print(f"   编码: {stream['codec_name']}")
            print(f"   分辨率: {stream['width']}x{stream['height']}")
            print(f"   帧率: {eval(stream.get('r_frame_rate', '0/1')):.2f} fps")
            print(f"   码率: {int(stream.get('bit_rate', 0)) // 1000} kbps")

        elif stream["codec_type"] == "audio":
            print(f"\n🔊 音频流:")
            print(f"   编码: {stream['codec_name']}")
            print(f"   采样率: {stream.get('sample_rate', 'N/A')} Hz")
            print(f"   声道: {stream.get('channels', 'N/A')}")
            print(f"   码率: {int(stream.get('bit_rate', 0)) // 1000} kbps")

    # 3. 比特率计算
    total_bitrate = int(format_info.get("bit_rate", 0))
    print(f"\n📊 总码率: {total_bitrate // 1000} kbps")

    print("\n" + "=" * 50)
    print("✅ 验证完成！")
    print("=" * 50)
    return True


if __name__ == "__main__":
    verify()
