# coding:utf-8
import cv2
import os

import dlib
import time


# http://blog.topspeedsnail.com/archives/6935

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


# 开始捕获视频
vid_cam = cv2.VideoCapture(0)

# 利用Haarcascade正面检测视频流中的目标
# face_detector = cv2.CascadeClassifier('./face_model/haarcascade_frontalface_default.xml')

# 每录入一张人脸的时候在这里写一个id，记住一点就是每个人的ID都不能相同。
face_id = 1

# 使用 Dlib 的正面人脸检测器 frontal_face_detector
detector = dlib.get_frontal_face_detector()

# Dlib 的 68点模型
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# 初始化样本人脸图像
count = 0
path = "./dataTest"
assure_path_exists(path)
win = dlib.image_window()

# 开始循环
while (True):

    # 捕获的视频帧
    _, image_frame = vid_cam.read()

    # 帧转换为灰度图
    # gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # 检测不同大小的帧，人脸矩形列表，返回四个值就是人脸位置的坐标
    # faces = face_detector.detectMultiScale(image_frame, 1.3, 5)

    # 使用 detector 检测器来检测图像中的人脸
    start = time.time()
    faces = detector(image_frame, 1)
    end = time.time()
    print("detector Time Spent: ", end - start)

    print("人脸数：", len(faces), )  # [im for im in faces]
    win.clear_overlay()
    win.set_image(image_frame)

    for i, d in enumerate(faces):
        print("第", i + 1, "个人脸的矩形框坐标：",
              "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())

        # cv2.rectangle(image_frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)

        start = time.time()
        # 使用predictor来计算面部轮廓
        shapes = predictor(image_frame, faces[i])
        # 'num_parts', 'part', 'parts', 'rect']
        # print(dir(shapes.parts))
        end = time.time()
        print("predictor Time Spent: ", end - start)

        # 绘制面部轮廓
        win.add_overlay(shapes)

        # 绘制矩阵轮廓
        win.add_overlay(faces)

    # cv2.imshow('frame', image_frame)

    # 停止录像，按“q”键至少100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # 如果拍摄的图像达到100，停止拍摄视频
    # elif count == 10:
    #     break

# 停止视频
dlib.hit_enter_to_continue()
