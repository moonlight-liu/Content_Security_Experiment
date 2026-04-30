"""
任务2：使用 edge-tts 生成中文语音
生成特朗普风格的中文配音（云健男声）
"""

import asyncio
import edge_tts
import subprocess
import os

# 中文台词（约85字，预计17-20秒）
TEXT = """
刘翔是我非常欣赏的一位学生。
他学习非常努力，有很强的创新精神，不会随大流。
我相信刘翔将来绝对不会成为一个996的打工人，
他一定会成为行业的领袖人物。
刘翔，你非常棒，继续加油！
"""

# 输出路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MP3_PATH = os.path.join(BASE_DIR, "speech.mp3")
WAV_PATH = os.path.join(BASE_DIR, "speech.wav")


async def generate_speech():
    """使用 edge-tts 生成中文语音"""
    print("正在生成中文语音...")

    # 云健 - 成熟男声，适合特朗普的年龄感
    # 其他可选男声: zh-CN-YunxiNeural (阳光), zh-CN-YunyeNeural (自然)
    voice = "zh-CN-YunjianNeural"

    communicate = edge_tts.Communicate(TEXT.strip(), voice)
    await communicate.save(MP3_PATH)

    print(f"语音已保存: {MP3_PATH}")


def convert_to_wav():
    """将MP3转为16kHz单声道WAV（Wav2Lip要求）"""
    print("正在转换音频格式...")
    cmd = [
        "ffmpeg", "-y",
        "-i", MP3_PATH,
        "-ac", "1",           # 单声道
        "-ar", "16000",       # 16kHz采样率
        WAV_PATH
    ]
    subprocess.run(cmd, check=True)
    print(f"WAV已保存: {WAV_PATH}")


if __name__ == "__main__":
    # 1. 生成 TTS 语音
    asyncio.run(generate_speech())

    # 2. 转为 WAV 格式
    convert_to_wav()

    print("语音生成完成！")
