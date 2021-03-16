# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np
import math


class FaceCrop:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
        self.headmarks = np.zeros((21, 2))
        self.shouldermarks = np.zeros((8, 2))

    def get_points(self, image):
        dets = self.detector(image, 1)
        if not dets:
            return None  # np.array([])

        points = np.zeros((19, 2))
        for k, d in enumerate(dets):
            shape = self.predictor(image, d)
            for i in range(17):
                points[i] = (shape.part(i).x, shape.part(i).y)
            points[17] = (shape.part(24).x, shape.part(24).y - 20)
            points[18] = (shape.part(19).x, shape.part(19).y - 20)

        return points

    def get_markers(self, image):
        h = image.shape[0]
        w = image.shape[1]
        dets = self.detector(image, 1)
        if not dets:
            return None

        points = np.zeros((19, 2))
        self.headmarks = np.zeros((21, 2))
        self.shouldermarks = np.zeros((8, 2))
        for k, d in enumerate(dets):
            shape = self.predictor(image, d)
            # 计算脸长度的三分之一大小，向上取整
            delta_y = math.ceil((shape.part(8).y - shape.part(19).y) / 2)
            # 计算脸宽度
            delta_x = shape.part(14).x - shape.part(2).x
            # print(delta_x, delta_y)
            # 脸
            for i in range(17):
                self.headmarks[i] = (shape.part(i).x, shape.part(i).y)
                points[i] = (shape.part(i).x, shape.part(i).y)
            points[17] = (shape.part(24).x, shape.part(24).y - 20)
            points[18] = (shape.part(19).x, shape.part(19).y - 20)
            # 额头
            # self.headmarks[0][1] -= 150
            # landmarks[17] = (shape.part(24).x, shape.part(24).y - 20)
            # landmarks[18] = (shape.part(19).x, shape.part(19).y - 20)

            self.headmarks[17] = (shape.part(16).x, max(0, shape.part(16).y - delta_y))
            self.headmarks[18] = (shape.part(24).x, max(0, shape.part(24).y - 20 - delta_y))
            self.headmarks[19] = (shape.part(19).x, max(0, shape.part(19).y - 20 - delta_y))
            self.headmarks[20] = (shape.part(0).x, max(0, shape.part(0).y - delta_y))
            # 脖子
            self.shouldermarks[0] = (shape.part(5).x, shape.part(5).y)
            self.shouldermarks[1] = (shape.part(5).x, min(h, shape.part(5).y + delta_y))
            # 肩膀
            self.shouldermarks[2] = (max(0, self.shouldermarks[1][0] - delta_x), self.shouldermarks[1][1])
            self.shouldermarks[3] = (self.shouldermarks[2][0], min(h, self.shouldermarks[2][1] + 100))
            # 脖子
            self.shouldermarks[7] = (shape.part(11).x, shape.part(11).y)
            self.shouldermarks[6] = (shape.part(11).x, min(h, shape.part(11).y + delta_y))
            # 肩膀
            self.shouldermarks[5] = (min(w, self.shouldermarks[6][0] + delta_x), self.shouldermarks[6][1])
            self.shouldermarks[4] = (self.shouldermarks[5][0], min(h, self.shouldermarks[5][1] + 100))

        return points

    def get_mask(self, image, points):
        # 和原始图像一样大小的0矩阵，作为mask
        mask = np.zeros(image.shape[:2], np.uint8)
        if points is None:  # points == np.array([]):
            return mask

        points = np.asarray(points, dtype=np.int32)
        points = np.array([points])
        # 在mask上将多边形区域填充为白色
        cv2.polylines(mask, points, True, (0, 255, 255))  # 描绘边缘
        cv2.fillPoly(mask, points, 255)  # 填充

        return mask

    def get_expand_mask(self, image):
        headmask = self.get_mask(image, self.headmarks)
        shouldermask = self.get_mask(image, self.shouldermarks)
        mask = cv2.bitwise_or(headmask, shouldermask)
        return mask
