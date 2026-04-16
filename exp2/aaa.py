import cv2
import numpy as np

# -------------------------------
# 1. 配置参数
# -------------------------------
config_path = "yolov2.cfg"  # 网络配置文件路径
weights_path = "yolov2.weights"  # 权重文件路径
classes_path = "coco.names"  # 类别文件路径
confidence_threshold = 0.5  # 置信度阈值
nms_threshold = 0.4  # 非极大值抑制阈值
input_width = 416  # 网络输入宽度
input_height = 416  # 网络输入高度

# 加载类别名称
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 生成随机颜色用于绘制不同类别的框
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

# -------------------------------
# 2. 加载待叠加的图像（支持透明通道），并缩小为0.5倍
# -------------------------------
overlay_path = r"a.png"
overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # 读取为BGRA
if overlay_img is None:
    print(f"警告：无法加载叠加图像 {overlay_path}，将不进行叠加")
    overlay_img = None
else:
    # 如果图像没有Alpha通道，添加一个全不透明的Alpha通道
    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
    # 缩放为原尺寸的0.5倍
    overlay_img = cv2.resize(overlay_img, (0, 0), fx=0.5, fy=0.5)


def overlay_image_left_center(background, overlay):
    """
    将 overlay 图像（BGRA）叠加到 background 的**左半边中间**位置，
    返回叠加后的图像。
    """
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay.shape[:2]

    # 左半边区域：宽度为 w_bg//2，高度为整个画面
    left_half_width = w_bg // 2

    # 计算水平位置：让图像在左半边内水平居中
    if w_ov <= left_half_width:
        x = (left_half_width - w_ov) // 2
    else:
        # 若图像宽度大于左半边，则靠左对齐（避免超出画面）
        x = 0

    # 垂直居中
    y = (h_bg - h_ov) // 2

    # 确保坐标不超出边界
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    # 计算有效区域（防止超出背景边界）
    x_end = min(x + w_ov, w_bg)
    y_end = min(y + h_ov, h_bg)
    w_ov_actual = x_end - x
    h_ov_actual = y_end - y

    if w_ov_actual <= 0 or h_ov_actual <= 0:
        return background  # 没有有效区域

    # 提取背景ROI
    roi = background[y:y_end, x:x_end]
    # 裁剪overlay到相同大小
    overlay_cropped = overlay[0:h_ov_actual, 0:w_ov_actual]

    # 分离Alpha通道
    overlay_bgr = overlay_cropped[:, :, :3]
    alpha = overlay_cropped[:, :, 3] / 255.0

    # Alpha混合
    for c in range(3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_bgr[:, :, c]

    # 写回背景
    background[y:y_end, x:x_end] = roi
    return background


# -------------------------------
# 3. 加载 YOLOv2 网络
# -------------------------------
net = cv2.dnn.readNet(weights_path, config_path)

# 若想使用 GPU（需编译 OpenCV 支持 CUDA）：
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 获取输出层名称
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -------------------------------
# 4. 打开摄像头
# -------------------------------
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

print("开始实时检测，按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # 5. 先叠加图像到帧（左半边中间）
    # -------------------------------
    if overlay_img is not None:
        frame = overlay_image_left_center(frame, overlay_img)

    # 此时 frame 已经是叠加后的图像，后续 YOLO 将基于此进行检测
    h, w = frame.shape[:2]

    # -------------------------------
    # 6. 构建输入 blob 并前向传播（基于叠加后的帧）
    # -------------------------------
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (input_width, input_height), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # -------------------------------
    # 7. 解析输出，收集检测结果
    # -------------------------------
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] * detection[4]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width_box = int(detection[2] * w)
                height_box = int(detection[3] * h)

                x = int(center_x - width_box / 2)
                y = int(center_y - height_box / 2)

                boxes.append([x, y, width_box, height_box])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # -------------------------------
    # 8. 非极大值抑制
    # -------------------------------
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # -------------------------------
    # 9. 在叠加后的帧上绘制检测结果
    # -------------------------------
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]].tolist()

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            text = f"{label}: {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv2.putText(
                frame,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    # 显示结果
    cv2.imshow("YOLOv2 Real-time Detection", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
