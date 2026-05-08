import cv2
import dlib
import os
from deepface import DeepFace

# 1. 初始化 dlib 的人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 2. 设置人脸库路径（用来做创新拓展的人脸识别）
db_path = "face_db"

# 3. 打开摄像头
cap = cv2.VideoCapture(0)

print("正在加载 Deepface 模型并打开摄像头，首次运行可能需要下载权重文件，请耐心等待...")

frame_count = 0  # 帧计数器
# 用于保存每 10 帧检测出的一次结果，避免画面刷新掉
current_emotion = "Waiting..."
current_age = "Waiting..."
current_name = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 获取灰度图供 dlib 使用
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # 【核心任务2要求】：每隔 10 帧利用 Deepface 检测一次
    if frame_count % 10 == 0 and len(faces) > 0:
        try:
            # (1) 属性分析：检测年龄、情感 (enforce_detection=False 防止没完全检测到脸时报错)
            analyze_result = DeepFace.analyze(
                img_path=frame,
                actions=["age", "emotion"],
                enforce_detection=False,
                silent=True,
            )
            # Deepface 有时返回列表，有时返回字典
            if isinstance(analyze_result, list):
                analyze_result = analyze_result[0]

            current_age = str(analyze_result.get("age", "N/A"))
            current_emotion = str(analyze_result.get("dominant_emotion", "N/A"))

            # (2) 创新拓展：人脸识别 (与 face_db 库进行比对)
            if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
                find_result = DeepFace.find(
                    img_path=frame,
                    db_path=db_path,
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(find_result, list):
                    find_result = find_result[0]

                # 如果 DataFrame 不为空，说明匹配到了库里的人
                if not find_result.empty:
                    # 获取匹配到的文件路径，比如 "face_db/2023312181003_Li_Zhen_Hong_Cyberspace Security_male.jpg"
                    matched_path = find_result["identity"][0]
                    filename = os.path.basename(matched_path)

                    # 现在文件名格式为: 学号_英文名_Cyberspace Security_性别.jpg
                    # 采用正则解析：学号在开头，性别在末尾，专业固定为 "Cyberspace Security"
                    fname_noext = os.path.splitext(filename)[0]
                    import re

                    pattern = re.compile(
                        r"^(?P<id>\d+)_(?P<name>.+)_(?P<major>Cyberspace Security)_(?P<gender>[^_]+)$"
                    )
                    m2 = pattern.match(fname_noext)
                    if m2:
                        stu_id = m2.group("id")
                        stu_name = m2.group("name").replace("_", " ")
                        stu_major = m2.group("major")
                        stu_gender = m2.group("gender")
                        current_name = stu_name

                        # 在终端输出识别信息（英文名）
                        print(
                            f"[{frame_count}帧] 🎓 识别成功: {stu_name} | 专业: {stu_major} | 性别: {stu_gender} | 学号: {stu_id} | 当前情绪: {current_emotion}"
                        )
                    else:
                        # 如果不匹配新格式，尝试使用旧的以防有遗留文件
                        name_parts = fname_noext.split("-")
                        if len(name_parts) >= 4:
                            stu_id = name_parts[0]
                            stu_name = name_parts[1]
                            stu_major = name_parts[2]
                            stu_gender = name_parts[3]
                            current_name = stu_name
                            print(
                                f"[{frame_count}帧] 🎓 识别成功(旧格式): {stu_name} | 专业: {stu_major} | 性别: {stu_gender} | 学号: {stu_id} | 当前情绪: {current_emotion}"
                            )

        except Exception as e:
            # 忽略没有检测到完整人脸时的报错
            pass

    # 【任务1结合】：遍历人脸绘制框和关键点
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # 画绿框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 提取并画出 68 个关键点
        landmarks = predictor(gray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # 在人脸框上方打印 Deepface 的检测结果（年龄、情绪、拼音/英文名）
        text_info = f"Age:{current_age} Emotion:{current_emotion}"
        cv2.putText(
            frame,
            text_info,
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Name:{current_name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.imshow("Task 2: Emotion & Age & Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
