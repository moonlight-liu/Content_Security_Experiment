import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm  # 用于显示进度条


def extract_features(dataset_path):
    """
    遍历UrbanSound8K数据集，提取音频的MFCC特征，并返回特征矩阵和标签矩阵
    """
    # 定义路径
    csv_path = os.path.join(dataset_path, "metadata", "UrbanSound8K.csv")
    audio_dir = os.path.join(dataset_path, "audio")

    print(f"正在读取标注文件: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到标注文件: {csv_path}，请确认 UrbanSound8K 目录是否在 exp2 下。"
        )
    df = pd.read_csv(csv_path)

    features = []
    labels = []

    print("开始提取音频特征 (这个过程可能需要几分钟，请耐心等待)...")

    # 遍历CSV文件中的每一行数据
    # 使用tqdm包装以显示进度条
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # 获取音频所在的fold文件夹和文件名
        fold_name = f"fold{row['fold']}"
        file_name = row["slice_file_name"]
        label = row["classID"]  # 获取分类标签 (0-9)

        file_path = os.path.join(audio_dir, fold_name, file_name)

        # 如果你删除了某些fold，找不到文件时就自动跳过
        if not os.path.exists(file_path):
            continue

        try:
            # 1. 加载音频文件
            # res_type='kaiser_fast' 可以加快重采样速度
            X, sample_rate = librosa.load(file_path, res_type="kaiser_fast")

            # 2. 提取MFCC特征
            # n_mfcc=40 是音频分类任务中常用的维度
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)

            # 3. 对时间轴求平均值，将2D的特征图压缩成1D的向量 (长度为40)
            # 这样做可以极大减少深度学习模型的参数量和训练时间
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            features.append(mfccs_scaled)
            labels.append(label)
        except Exception as e:
            print(f"解析文件 {file_path} 失败: {e}")
            continue

    # 将列表转换为 numpy 数组，方便后续输入给深度学习模型
    features = np.array(features)
    labels = np.array(labels)

    return features, labels


if __name__ == "__main__":
    # 使用脚本所在目录作为基准，避免在不同终端路径下运行时报错
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "UrbanSound8K")

    X, y = extract_features(DATASET_PATH)

    print(f"\n特征提取完成！共提取了 {len(X)} 条音频数据。")
    print(f"特征矩阵 X 的形状: {X.shape}")  # 预期形状: (样本数, 40)
    print(f"标签矩阵 y 的形状: {y.shape}")  # 预期形状: (样本数,)

    # 将提取好的特征保存到本地，这样下一步训练模型时就不需要重新提取了！
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)
    print("特征已成功保存为 X_features.npy 和 y_labels.npy！")
