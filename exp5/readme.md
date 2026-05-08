# 图像伪造与检测实验项目 (Content Security Lab)

本项目基于 Python 3 实现了从人脸基础特征提取、属性分析到图像伪造及其检测的完整链路实验。

## 🚀 实验功能模块

### 1. 实时人脸识别与 68 关键点检测
*   **代码文件**: `task1_landmarks.py`
*   **技术栈**: OpenCV + Dlib (HOG + SVM)
*   **功能**: 调用摄像头，实时检测人脸位置并绘制 68 个面部关键点（Landmarks）。

### 2. 人脸属性分析与班级身份识别 (创新拓展)
*   **代码文件**: `task2_emotion_recognition.py`
*   **技术栈**: DeepFace + Dlib
*   **功能**: 
    *   每隔 10 帧分析一次人脸的年龄、性别和情绪（如：Neutral, Happy, Surprise）。
    *   **创新点**: 结合自定义人脸库 `face_db`，实现了基于文件名的身份自动解析识别，支持识别学生姓名、学号及专业。
    *   **识别格式**: `学号_姓名_专业_性别.jpg`

### 3. 人脸伪造 (Face Swap)
*   **代码文件**: `task3_faceswap.py`
*   **技术栈**: Dlib + OpenCV (Affine Transform + Seamless Cloning)
*   **功能**: 将源图（Source）的面部特征提取并对齐旋转，通过无缝克隆技术融合到目标图（Target）中，生成高质量的伪造人脸图像（如：雷军换脸图）。

### 4. 人脸伪造检测 (Face-X-Ray)
*   **代码文件**: `task4_run.py` (驱动脚本) + `detect_video.py` (核心算法)
*   **技术栈**: PyTorch + HRNet + Face-X-Ray 预训练模型
*   **功能**: 针对任务 3 生成的伪造图进行“X光”检测。通过识别图像融合边界（Blending Boundary），以热力图形式展现造假痕迹，并给出 `real/fake` 的判定结果。

## 📂 关键环境与模型配置

| 文件/目录 | 说明 |
| :--- | :--- |
| `shape_predictor_68_face_landmarks.dat` | Dlib 68点关键点检测模型 |
| `face_db/` | 存放班级同学照片的数据库，用于身份识别 |
| `HRNet/pretrained/` | 存放 HRNet 骨干网络预训练权重 |
| `result/best_model.pth.tar` | Face-X-Ray 核心检测模型权重 |
| `fake_result.jpg` | 任务 3 生成的待检测伪造图片 |

## 🛠️ 安装与运行

1.  **安装依赖**:
    ```bash
    pip install opencv-python dlib deepface torch torchvision tqdm
    ```
2.  **运行检测**:
    *   执行任务 1/2 进行实时交互。
    *   执行任务 3 生成伪造图像。
    *   执行 `python task4_run.py` 自动完成 X-Ray 检测并生成结果视频。

## 📊 实验结论
通过 Face-X-Ray 检测，伪造图像在边缘融合处表现出明显的异常响应（白色高亮区域），模型成功判定为 **Fake**。实验证明了基于图像融合边界的检测算法在应对传统换脸技术时具有极高的有效性和鲁棒性。

