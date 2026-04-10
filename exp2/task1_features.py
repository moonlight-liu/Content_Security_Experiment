import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 解决matplotlib中文显示问题（如果在图表中需要显示中文）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows用黑体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def extract_audio_features(file_path):
    """
    提取并绘制音频的波形图、声谱图和MFCC特征图
    """
    print(f"正在加载音频文件: {file_path} ...")

    # 1. 加载音频文件
    # sr=None 表示保留原始采样率，如果不设置默认为 22050
    data, sampling_rate = librosa.load(file_path, sr=22050)

    # 检查音频长度是否大于10秒
    duration = librosa.get_duration(y=data, sr=sampling_rate)
    print(f"音频加载成功！采样率: {sampling_rate} Hz, 时长: {duration:.2f} 秒")
    if duration < 10.0:
        print("【警告】实验要求音频长度不小于10s，建议更换更长的音频！")

    # 创建一个大的画布，包含3个子图
    plt.figure(figsize=(14, 15))

    # ==========================================
    # 任务1.1: 绘制时域波形图 (Waveform)
    # ==========================================
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(data, sr=sampling_rate)
    plt.title("音频时域波形图 (Waveform)", fontsize=14)
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅值")

    # ==========================================
    # 任务1.2: 绘制声谱图 (Spectrogram)
    # ==========================================
    plt.subplot(3, 1, 2)
    # STFT: 短时傅里叶变换，将一维时域信号转换为二维时频信号
    D = librosa.stft(data)
    # 将幅度值转换为分贝(dB)单位，方便人眼观察
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # 绘制声谱图，y轴为线性频率
    librosa.display.specshow(S_db, sr=sampling_rate, x_axis="time", y_axis="linear")
    plt.colorbar(format="%+2.0f dB")  # 添加颜色条
    plt.title("线性频率声谱图 (Linear-frequency Spectrogram)", fontsize=14)

    # ==========================================
    # 任务1.3: 绘制MFCC特征图
    # ==========================================
    plt.subplot(3, 1, 3)
    # 提取MFCC特征，通常保留n_mfcc=20或13维
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=20)
    # 绘制MFCC图
    librosa.display.specshow(mfccs, sr=sampling_rate, x_axis="time")
    plt.colorbar()
    plt.title("梅尔频率倒谱系数 (MFCC)", fontsize=14)
    plt.ylabel("MFCC 系数")

    # 自动调整子图间距并显示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    audio_path = "exp2/marmixer-see-you-later-203103.wav"
    extract_audio_features(audio_path)
