import cv2
import dlib
import numpy as np

# 1. 初始化 dlib 人脸检测器和关键点提取器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(img):
    """获取图像中第一张人脸的 68 个关键点"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    # 只取检测到的第一张脸
    return np.array([[p.x, p.y] for p in predictor(gray, faces[0]).parts()])


def create_face_mask(img_shape, landmarks):
    """根据面部关键点外轮廓生成掩膜 (Mask)"""
    # 获取关键点构成的凸包（面部外轮廓）
    hull = cv2.convexHull(landmarks)
    # 创建与原图一样大小的全黑画布
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    # 将面部凸包区域填充为纯白 (255)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def swap_faces(target_img_path, source_img_path):
    # 读取图片
    img_target = cv2.imread(target_img_path)
    img_source = cv2.imread(source_img_path)

    if img_target is None or img_source is None:
        print("图片读取失败，请检查 target.jpg 和 source.jpg 是否存在！")
        return None

    # 第一步：检测目标图像和源图像的面部关键点 (对应课件 Detect)
    landmarks_target = get_landmarks(img_target)
    landmarks_source = get_landmarks(img_source)

    if landmarks_target is None or landmarks_source is None:
        print("未能检测到人脸，请换清晰的正脸图片！")
        return None

    # 第二步：仿射变换与面部生成 (对应课件 Manipulate)
    # 计算从 源人脸 到 目标人脸 的仿射变换矩阵（旋转、平移、缩放的对齐）
    M, _ = cv2.estimateAffinePartial2D(landmarks_source, landmarks_target)
    # 将源图像的脸扭曲/变形，完全贴合目标图像的脸部位置
    warped_source = cv2.warpAffine(
        img_source, M, (img_target.shape[1], img_target.shape[0])
    )

    # 获取融合所需的掩膜 (以目标图像的脸型为准)
    mask = create_face_mask(img_target.shape, landmarks_target)

    # 将掩膜稍微腐蚀(缩小)一点，防止把背景边缘也抠进去，让融合更自然
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    # 第三步：图像融合 (对应课件 Blend)
    # 获取掩膜的边界矩形，以计算中心点
    x, y, w, h = cv2.boundingRect(mask)
    center = (x + w // 2, y + h // 2)

    try:
        # 使用 OpenCV 的无缝克隆算法 (Seamless Clone) 完美融合光照和肤色
        result = cv2.seamlessClone(
            warped_source, img_target, mask, center, cv2.NORMAL_CLONE
        )
        return result
    except Exception as e:
        print("融合失败，可能是掩膜边缘超出了图像边界：", e)
        return None


if __name__ == "__main__":
    print("正在进行 AI 换脸伪造，请稍等...")

    # 定义输入图片路径
    target_path = "target.jpg"  # 身体
    source_path = "source.jpg"  # 脸

    output_img = swap_faces(target_path, source_path)

    if output_img is not None:
        # 依次显示结果
        cv2.imshow("1. Target (Body)", cv2.imread(target_path))
        cv2.imshow("2. Source (Face)", cv2.imread(source_path))
        cv2.imshow("3. Fake Face (Swapped)", output_img)

        print("换脸成功！按键盘任意键保存图片并退出。")
        cv2.waitKey(0)
        cv2.imwrite("fake_result.jpg", output_img)  # 保存造假生成的图片
        cv2.destroyAllWindows()
