import os
import cv2
import xml.etree.ElementTree as ET
import random

# 你的路径（按当前文件所在目录拼接，避免运行目录不同导致找不到文件）
base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Train", "JPEGImages")
xml_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Train", "Annotations")
out_pos_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Train", "pos_64x128")
os.makedirs(out_pos_dir, exist_ok=True)

count = 0

for xml_name in os.listdir(xml_dir):
    xml_path = os.path.join(xml_dir, xml_name)
    stem = os.path.splitext(xml_name)[0]
    img_path = None
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = os.path.join(img_dir, stem + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"跳过: 找不到对应图片 {stem}")
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img = cv2.imread(img_path)
    if img is None:
        print(f"跳过: 图片读取失败 {img_path}")
        continue

    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls != "person":
            continue

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # 抠出行人
        person = img[ymin:ymax, xmin:xmax]
        # 缩成标准 64x128
        person = cv2.resize(person, (64, 128))

        cv2.imwrite(os.path.join(out_pos_dir, f"pos_{count}.png"), person)
        count += 1

print("正样本生成完成！数量：", count)

img_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Train", "JPEGImages")
out_neg_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Train", "neg_64x128")
os.makedirs(out_neg_dir, exist_ok=True)

count = 0
need = 2000  # 要多少负样本

done = False
for fname in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir, fname))
    if img is None:
        continue
    h, w = img.shape[:2]
    if w < 64 or h < 128:
        continue

    for _ in range(10):
        x = random.randint(0, w - 64)
        y = random.randint(0, h - 128)
        patch = img[y : y + 128, x : x + 64]
        cv2.imwrite(os.path.join(out_neg_dir, f"neg_{count}.png"), patch)
        count += 1
        if count >= need:
            done = True
            break
    if done:
        break

print("负样本生成完成！数量：", count)
