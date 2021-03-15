# coding=utf-8
from .face_detect import FaceDetect
from .face_seg import FaceSeg
# from yoloFace.face_detector import face_detect
from facecrop.faceCrop import FaceCrop
import numpy as np
import cv2
import time
import dlib


class Preprocess:
    def __init__(self, device='cpu', detector='dlib'):
        self.detect = FaceDetect(device, detector)  # device = 'cpu' or 'cuda', detector = 'dlib' or 'sfd'
        self.segment = FaceSeg()
        self.face_crop = FaceCrop()
        # 使用 Dlib 的正面人脸检测器 frontal_face_detector
        # self.detector = dlib.get_frontal_face_detector()
        # Dlib 的 68点模型 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
        self.rect = []

    def process_optimize(self, image):
        landmarks = self.face_crop.get_markers(image)
        if landmarks is None:
            print("cannot detect any landmarks")
            return

        # # get face from landmarks
        # face, self.rect = self.crop(image, landmarks)
        # face = face[self.rect[0]:self.rect[1] + 1, self.rect[2]:self.rect[3] + 1]
        # # get pure face mask without background
        # mask = self.face_crop.get_mask(image, landmarks)
        # mask = mask[self.rect[0]:self.rect[1] + 1, self.rect[2]:self.rect[3] + 1]

        # gzx
        face, self.rect = self.crop_optimize(image, landmarks)
        mask = self.segment.get_mask(face)  # segment mask
        # mask = self.crop_mask_stretch(image, landmarks, [face.shape[0], face.shape[1]])

        # xinbi
        # mask = self.face_crop.get_expand_mask(image)[:, :, np.newaxis]
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/mask_before_crop.png", mask)
        # mask = self.crop_mask(mask, [face.shape[0], face.shape[1]])

        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/face.png", face)
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/mask_after_crop.png", mask)
        return np.dstack((face, mask))

    def process_origin(self, image):
        start = time.time()
        face_info = self.detect.align(image)  # 0.822
        end = time.time()
        print("face_align: ", end - start)
        if face_info is None:
            return None
        image_align, landmarks_align = face_info

        start = time.time()
        face = self.__crop(image_align, landmarks_align)  # 0.0005
        end = time.time()
        print("face_detect: ", end - start)

        start = time.time()
        mask = self.segment.get_mask(face)  # 0.301
        end = time.time()
        print("face_segment: ", end - start)
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/image_align.png", image_align)
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/face.png", face)
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/mask.png", mask)
        return np.dstack((face, mask))

    @staticmethod
    def __crop(image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        # make sure the output image is square
        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w - 1)
        right_white = left_white + (right - left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h - 1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1,
                                                                             left:right + 1].copy()
        return image_crop

    def crop_optimize(self, image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        # make sure the output image is square
        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w - 1)
        right_white = left_white + (right - left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h - 1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1,
                                                                             left:right + 1].copy()
        return image_crop, [[top, bottom, left, right], [top_white, bottom_white, left_white, right_white]]

    def crop_track(self, image, boxes):
        landmarks_top = boxes[0]
        landmarks_bottom = boxes[1]
        landmarks_left = boxes[2]
        landmarks_right = boxes[3]

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        # make sure the output image is square
        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w - 1)
        right_white = left_white + (right - left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h - 1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1,
                                                                             left:right + 1].copy()
        return image_crop, [[top, bottom, left, right], [top_white, bottom_white, left_white, right_white]]

    def crop_mask_stretch(self, image, landmarks, size):
        mask = self.face_crop.get_mask(image, landmarks)

        top = self.rect[0][0]
        bottom = self.rect[0][1]
        left = self.rect[0][2]
        right = self.rect[0][3]
        top_white = self.rect[1][0]
        bottom_white = self.rect[1][1]
        left_white = self.rect[1][2]
        right_white = self.rect[1][3]

        landmarks_top = int(np.min(landmarks[:, 1]))
        landmarks_bottom = int(np.max(landmarks[:, 1]))
        landmarks_left = int(np.min(landmarks[:, 0]))
        landmarks_right = int(np.max(landmarks[:, 0]))
        mid = (landmarks_top + landmarks_bottom) // 2
        hair = int(mid - (landmarks_bottom - landmarks_top) * 0.5)

        mask[landmarks_bottom:bottom, left:right] = np.ones((bottom - landmarks_bottom, right - left),
                                                            dtype=np.uint8) * 255
        mask[hair:landmarks_top, landmarks_left:landmarks_right] = np.ones(
            (landmarks_top - hair, landmarks_right - landmarks_left), dtype=np.uint8) * 255
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/mask_before_crop.png", mask)

        mask_crop = np.zeros((size[0], size[1], 1), np.uint8)
        mask_crop[top_white:bottom_white + 1, left_white:right_white + 1] = mask[:, :, np.newaxis][top:bottom + 1,
                                                                            left:right + 1].copy()
        return mask_crop

    def crop_mask(self, mask, size):
        top = self.rect[0][0]
        bottom = self.rect[0][1]
        left = self.rect[0][2]
        right = self.rect[0][3]
        top_white = self.rect[1][0]
        bottom_white = self.rect[1][1]
        left_white = self.rect[1][2]
        right_white = self.rect[1][3]

        mask_crop = np.zeros((size[0], size[1], 1), np.uint8)
        mask_crop[top_white:bottom_white + 1, left_white:right_white + 1] = mask[top:bottom + 1,
                                                                            left:right + 1].copy()
        return mask_crop
