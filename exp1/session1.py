import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 定义HOG提取器 (OpenCV自带现成的，不需要像PPT里纯手工算，除非老师特别要求)
hog = cv2.HOGDescriptor()


def load_data(pos_dir, neg_dir):
    labels = []
    features = []
    # 读取正样本 (行人)
    for file in os.listdir(pos_dir):
        img = cv2.imread(os.path.join(pos_dir, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 128))  # 统一尺寸
        hist = hog.compute(img)
        features.append(hist.flatten())
        labels.append(1)  # 1表示行人

    # 读取负样本 (非行人)
    for file in os.listdir(neg_dir):
        img = cv2.imread(os.path.join(neg_dir, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 128))
        hist = hog.compute(img)
        features.append(hist.flatten())
        labels.append(0)  # 0表示非行人

    return np.array(features), np.array(labels)


# 2. 加载数据
X, y = load_data("path_to_positives", "path_to_negatives")

# 3. 划分训练集和测试集 (参考PPT第22页)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. 训练SVM分类器
svm_model = SVC(kernel="linear")  # 或者用 KNN: KNeighborsClassifier(n_neighbors=5)
svm_model.fit(X_train, y_train)

# 5. 预测与评估
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 进阶：使用滑动窗口(Sliding Window)在整张大图上框出行人（可选，但加上会很高分）
