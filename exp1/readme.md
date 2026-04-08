# 内容安全实验一：图像处理与分析 说明文档 

## 一、 实验运行环境

本实验代码基于 Python 3.10 编写，运行前请确保安装以下依赖库：
pip install numpy opencv-python scikit-learn ultralytics

## 二、 包含文件与运行步骤

1. 任务一：HOG + SVM 行人目标检测
核心文件：session1.py
辅助文件：makedata.py, make_testdata.py（用于对原 XML 标注数据进行清洗、抠图与随机裁剪，生成 64x128 标准正负样本。若样本已就绪，可无需重复运行此两项）。
运行指令：python session1.py
运行现象：控制台将输出特征维度、耗时及精准度分析（Classification Report）；同时会弹出一张包含检测结果（绿色 Person / 红色 Not Person）的图片验证窗口（按任意键即可关闭）。
2. 任务二：YOLOv8 目标检测本地部署
核心文件：任务二代码文件（如：session2.py）
运行指令：python session2.py
运行现象：程序具备 Fallback 容错机制，将自动加载 yolov8n.pt 对本地图库进行推理。控制台与生成的 runs/dog_predict_log.txt 将同步输出各图片的检测目标数量。最后会弹窗展示首张被成功画好边界框、置信度与类别（dog）的图像。
