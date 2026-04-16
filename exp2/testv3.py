import cv2
from ultralytics import YOLO

# 1. 加载 YOLOv3u 预训练模型（首次运行会自动下载）
model = YOLO("yolov3u.pt")

# 2. 打开默认摄像头（索引0）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

print("按 'q' 键退出...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 进行推理（可调整 conf 置信度阈值）
    results = model(frame, conf=0.5)

    # 4. 在画面上绘制检测结果
    annotated_frame = results[0].plot()

    # 5. 显示画面
    cv2.imshow("YOLOv3u 实时检测", annotated_frame)

    # 按 'q' 退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
