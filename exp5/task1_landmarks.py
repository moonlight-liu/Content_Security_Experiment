import cv2
import dlib

# 1. 初始化 dlib 的正脸检测器 (基于HOG+SVM)
detector = dlib.get_frontal_face_detector()

# 2. 初始化人脸关键点提取器，加载68点模型
# 注意：请确保该 dat 文件与当前 python 代码在同一目录下
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3. 打开摄像头 (参数0表示电脑内置的默认摄像头)
cap = cv2.VideoCapture(0)

print("正在打开摄像头，请稍等... (在视频窗口中按 'q' 键退出)")

while True:
    # 4. 按帧读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面，请检查设备！")
        break

    # 为了加快检测速度和提高准确度，通常将彩色画面转换为灰度图供 dlib 检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. 检测灰度图中的所有人物脸部
    faces = detector(gray)

    # 6. 遍历画面中检测到的每一张人脸
    for face in faces:
        # 获取人脸矩形框的左上角和右下角坐标
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # 用绿色线条画出人脸边框 (BGR格式: 0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 7. 检测该人脸区域内的 68 个关键点
        landmarks = predictor(gray, face)

        # 8. 遍历这 68 个关键点，并在画面上绘制出来
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # 用红色实心小圆圈标出关键点 (BGR格式: 0, 0, 255)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # 9. 实时显示绘制好人脸框和关键点的画面
    cv2.imshow("Task 1: Face Landmarks Detection", frame)

    # 10. 等待键盘输入，延时 1ms。如果按下 'q' 键则跳出循环结束程序
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 11. 释放摄像头并销毁所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()
