import cv2
import numpy as np
import os
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# ================= 1. 定义路径与参数 =================
# 按当前文件位置自动拼接，避免手动改绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_POS_DIR = os.path.join(BASE_DIR, "dataset", "INRIAPerson", "Train", "pos_64x128")
TRAIN_NEG_DIR = os.path.join(BASE_DIR, "dataset", "INRIAPerson", "Train", "neg_64x128")
TEST_POS_DIR = os.path.join(BASE_DIR, "dataset", "INRIAPerson", "Test", "pos_64x128")
TEST_NEG_DIR = os.path.join(BASE_DIR, "dataset", "INRIAPerson", "Test", "neg_64x128")

# 初始化 OpenCV 默认的 HOG 描述符 (默认 winSize 就是 64x128)
hog = cv2.HOGDescriptor()


# ================= 2. 定义加载数据并提取HOG特征的函数 =================
def load_data_and_extract_features(pos_dir, neg_dir):
    features = []
    labels = []

    print("[INFO] 开始提取正样本 HOG 特征...")
    # 读取正样本 (行人)，标签设为 1
    for file in os.listdir(pos_dir):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(pos_dir, file)
            # 灰度图读取即可，HOG计算主要依赖灰度梯度
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 确保尺寸是 64x128
            img = cv2.resize(img, (64, 128))
            # 计算 HOG 特征并展平为一维向量
            hist = hog.compute(img)
            features.append(hist.flatten())
            labels.append(1)

    print("[INFO] 开始提取负样本 HOG 特征...")
    # 读取负样本 (非行人)，标签设为 0
    for file in os.listdir(neg_dir):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(neg_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (64, 128))
            hist = hog.compute(img)
            features.append(hist.flatten())
            labels.append(0)

    # 转为 numpy 数组格式供 sklearn 使用
    return np.array(features), np.array(labels)


# ================= 3. 主程序 =================
if __name__ == "__main__":
    start_time = time.time()

    # 1. 提取训练集特征
    X_train, y_train = load_data_and_extract_features(TRAIN_POS_DIR, TRAIN_NEG_DIR)
    print(
        f"[INFO] 训练集特征提取完成！共得到 {X_train.shape[0]} 个样本，每个样本特征维度为 {X_train.shape[1]}"
    )

    # 2. 提取测试集特征
    X_test, y_test = load_data_and_extract_features(TEST_POS_DIR, TEST_NEG_DIR)
    print(
        f"[INFO] 测试集特征提取完成！共得到 {X_test.shape[0]} 个样本，每个样本特征维度为 {X_test.shape[1]}"
    )

    # 3. 基于 SVM 训练分类模型 (使用线性核函数最适合 HOG 特征)
    print("[INFO] 正在训练 SVM 模型，请稍候...")
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train, y_train)

    # 4. 输出分类效果
    print("[INFO] 正在测试集上进行预测...")
    y_pred = svm_model.predict(X_test)

    print("\n================ 分类结果 (Classification Report) ================")
    # 打印精度指标 (参考 PPT 第22页)
    report = classification_report(
        y_test, y_pred, target_names=["Non-Person (0)", "Person (1)"]
    )
    print(report)
    print("================================================================")

    end_time = time.time()
    print(f"[INFO] 程序运行总耗时: {end_time - start_time:.2f} 秒")

    # ================= 5. 附加：单张图片可视化验证 (用于写实验报告截图) =================
    print("\n[INFO] 正在进行单张图片可视化验证...")
    # 从测试集随机挑一张正样本做可视化
    test_img_name = os.listdir(TEST_POS_DIR)[0]
    test_img_path = os.path.join(TEST_POS_DIR, test_img_name)

    # 读取彩色图用于显示
    display_img = cv2.imread(test_img_path)
    # 转灰度用于算特征
    gray_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (64, 128))

    # 提取单张图片特征并预测
    single_feature = hog.compute(gray_img).flatten().reshape(1, -1)
    prediction = svm_model.predict(single_feature)[0]

    # 在图片上画结果
    label_text = "Person" if prediction == 1 else "Not Person"
    color = (0, 255, 0) if prediction == 1 else (0, 0, 255)  # 绿色对，红色错
    cv2.putText(
        display_img, label_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
    )

    # 显示图片 (按任意键关闭)
    cv2.imshow("HOG + SVM Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
