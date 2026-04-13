from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # 自动下载最基础的YOLOv8模型
model.predict(source="0", show=True)  # source="0" 调用本地摄像头，并实时显示画面
