import os
import sys
import time

import cv2
from ultralytics import YOLO


# ================= 1. 数据集路径与参数 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "DOG.v2i.yolov8", "test", "images")

WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "dog_train", "weights", "best.pt")
FALLBACK_MODEL = "yolov8n.pt"
LOG_TXT = os.path.join(BASE_DIR, "runs", "dog_predict_log.txt")
PREDICT_DIR = os.path.join(BASE_DIR, "runs", "dog_predict")
PREDICT_NAME = "predictions"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def predict_image(
    model, source, output_dir, confidence_threshold=0.25, iou_threshold=0.45
):
    results = model.predict(
        source=source,
        conf=confidence_threshold,
        iou=iou_threshold,
        save=True,
        project=output_dir,
        name=PREDICT_NAME,
        exist_ok=True,
    )

    for result in results:
        boxes = result.boxes
        image_name = os.path.basename(str(getattr(result, "path", source)))
        print(f"[INFO] {image_name} 检测到 {len(boxes)} 个目标")

    return results


def find_trained_weights(root_dir):
    candidates = []
    for current_root, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.lower() in ("best.pt", "last.pt"):
                candidates.append(os.path.join(current_root, file_name))

    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


if __name__ == "__main__":
    start_time = time.time()

    os.makedirs(PREDICT_DIR, exist_ok=True)

    with open(LOG_TXT, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print("[INFO] 正在加载检测模型...")
            trained_weights = find_trained_weights(os.path.join(BASE_DIR, "runs"))
            if trained_weights is not None:
                print(f"[INFO] 使用训练好的权重: {trained_weights}")
                model = YOLO(trained_weights)
            else:
                print("[WARN] 没有找到任何训练权重 best.pt/last.pt。")
                print(
                    "[WARN] 这通常表示你还没有执行过训练脚本，data.yaml 只是数据配置，不会自动生成权重。"
                )
                print(f"[WARN] 改用基础模型: {FALLBACK_MODEL}")
                model = YOLO(FALLBACK_MODEL)

            test_images = [
                file_name
                for file_name in os.listdir(TEST_IMAGE_DIR)
                if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if not test_images:
                raise FileNotFoundError(f"测试集图片为空: {TEST_IMAGE_DIR}")

            print(f"[INFO] 共找到 {len(test_images)} 张测试图像，开始批量预测...")
            results = predict_image(model, TEST_IMAGE_DIR, PREDICT_DIR)

            if results:
                print(
                    f"[INFO] 预测结果已保存到: {os.path.join(PREDICT_DIR, PREDICT_NAME)}"
                )
                first_annotated_img = results[0].plot()
                if first_annotated_img is not None:
                    cv2.imshow("YOLO Dog Detection", first_annotated_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    end_time = time.time()
    print(f"[INFO] 程序运行总耗时: {end_time - start_time:.2f} 秒")
