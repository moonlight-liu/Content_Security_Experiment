import os
import sys
import time

from ultralytics import YOLO


# ================= 1. 路径与参数 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "DOG.v2i.yolov8", "test", "images")

MODEL_NAME = "yolov8n.pt"
LOG_TXT = os.path.join(BASE_DIR, "runs", "coco_dog_demo_log.txt")
REPORT_TXT = os.path.join(BASE_DIR, "runs", "coco_dog_demo_report.txt")
PREDICT_DIR = os.path.join(BASE_DIR, "runs", "coco_dog_demo")
PREDICT_NAME = "predictions"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


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


def get_class_id_by_name(model, class_name):
    for class_id, name in model.names.items():
        if str(name).lower() == class_name.lower():
            return class_id
    return None


def write_demo_report(report_path, model_name, dog_class_id, image_count, results):
    total_detections = 0
    images_with_dog = 0
    max_confidence = 0.0

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("COCO 预训练模型 dog 类别演示报告\n")
        report_file.write(f"模型: {model_name}\n")
        report_file.write(f"dog 类别 ID: {dog_class_id}\n")
        report_file.write(f"测试图像数量: {image_count}\n")
        report_file.write("每张图像的 dog 检测结果:\n")

        for result in results:
            image_name = os.path.basename(str(getattr(result, "path", "unknown")))
            boxes = result.boxes
            count = len(boxes)

            if count > 0:
                images_with_dog += 1
                total_detections += count
                if getattr(boxes, "conf", None) is not None:
                    confidences = boxes.conf.tolist()
                    if confidences:
                        max_confidence = max(max_confidence, max(confidences))

            report_file.write(f"- {image_name}: dog 检测数量 = {count}\n")

        report_file.write("\n汇总:\n")
        report_file.write(f"- 含 dog 的图像数量: {images_with_dog}\n")
        report_file.write(f"- dog 总检测数量: {total_detections}\n")
        report_file.write(f"- 最高置信度: {max_confidence:.4f}\n")

    print(f"[INFO] 演示报告已保存到: {report_path}")


if __name__ == "__main__":
    start_time = time.time()

    os.makedirs(PREDICT_DIR, exist_ok=True)

    with open(LOG_TXT, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print("[INFO] 正在加载 COCO 预训练模型...")
            model = YOLO(MODEL_NAME)

            dog_class_id = get_class_id_by_name(model, "dog")
            if dog_class_id is None:
                raise RuntimeError("在当前 COCO 模型中没有找到 dog 类别。")

            test_images = [
                file_name
                for file_name in os.listdir(TEST_IMAGE_DIR)
                if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not test_images:
                raise FileNotFoundError(f"测试集图片为空: {TEST_IMAGE_DIR}")

            print(f"[INFO] 找到 dog 类别 ID: {dog_class_id}")
            print(f"[INFO] 共找到 {len(test_images)} 张测试图像，开始批量演示预测...")

            results = model.predict(
                source=TEST_IMAGE_DIR,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=[dog_class_id],
                save=True,
                project=PREDICT_DIR,
                name=PREDICT_NAME,
                exist_ok=True,
            )

            write_demo_report(
                REPORT_TXT,
                MODEL_NAME,
                dog_class_id,
                len(test_images),
                results,
            )
            print("[INFO] 已完成 COCO dog 类别演示预测")
            print(f"[INFO] 预测结果目录: {os.path.join(PREDICT_DIR, PREDICT_NAME)}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    end_time = time.time()
    print(f"[INFO] 程序运行总耗时: {end_time - start_time:.2f} 秒")
