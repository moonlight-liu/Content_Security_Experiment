import os
import cv2
import xml.etree.ElementTree as ET
import random

# 按当前文件所在目录拼接路径
base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Test", "JPEGImages")
xml_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Test", "Annotations")
out_pos_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Test", "pos_64x128")
out_neg_dir = os.path.join(base_dir, "dataset", "INRIAPerson", "Test", "neg_64x128")

os.makedirs(out_pos_dir, exist_ok=True)
os.makedirs(out_neg_dir, exist_ok=True)

# ================= 1. 生成测试正样本 =================
pos_count = 0

for xml_name in os.listdir(xml_dir):
    if not xml_name.lower().endswith(".xml"):
        continue

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

    h, w = img.shape[:2]

    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls != "person":
            continue

        bndbox = obj.find("bndbox")
        xmin = max(0, int(bndbox.find("xmin").text))
        ymin = max(0, int(bndbox.find("ymin").text))
        xmax = min(w, int(bndbox.find("xmax").text))
        ymax = min(h, int(bndbox.find("ymax").text))

        if xmax <= xmin or ymax <= ymin:
            continue

        person = img[ymin:ymax, xmin:xmax]
        person = cv2.resize(person, (64, 128))

        cv2.imwrite(os.path.join(out_pos_dir, f"pos_{pos_count}.png"), person)
        pos_count += 1

print("测试集正样本生成完成！数量：", pos_count)

# ================= 2. 生成测试负样本 =================
# 用测试图片随机裁剪背景块，数量与正样本一致，保持测试集平衡
neg_need = pos_count
neg_count = 0
done = False

for fname in os.listdir(img_dir):
    img_path = os.path.join(img_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    if w < 64 or h < 128:
        continue

    for _ in range(10):
        x = random.randint(0, w - 64)
        y = random.randint(0, h - 128)
        patch = img[y : y + 128, x : x + 64]
        cv2.imwrite(os.path.join(out_neg_dir, f"neg_{neg_count}.png"), patch)
        neg_count += 1

        if neg_count >= neg_need:
            done = True
            break

    if done:
        break

print("测试集负样本生成完成！数量：", neg_count)
