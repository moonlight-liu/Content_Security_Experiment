from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CFG_PATH = BASE_DIR / "yolov2-tiny.cfg"
WEIGHTS_PATH = BASE_DIR / "yolov2-tiny.weights"
NAMES_PATH = BASE_DIR / "coco.names"

if not CFG_PATH.exists():
    raise FileNotFoundError(f"未找到配置文件: {CFG_PATH}")
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"未找到权重文件: {WEIGHTS_PATH}")
if not NAMES_PATH.exists():
    raise FileNotFoundError(f"未找到类别文件: {NAMES_PATH}")

# 1. 加载 YOLOv2-tiny 模型
net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(WEIGHTS_PATH))
with open(NAMES_PATH, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(
        "无法打开摄像头，请检查设备是否被占用，或尝试更换 VideoCapture 索引。"
    )

print("YOLOv2-tiny 已启动。尝试露出你的对抗补丁...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    # 图像预处理
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 我们只看“人”（Person在COCO数据集里通常是ID 0）
            if confidence > 0.3 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("Adversarial Test (YOLOv2-tiny)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
