#-*-coding:utf-8-*-
# from yoloface import yoloFace
import cv2
import os
import torch
import time
import argparse
from glob import glob
from Track.pysot.pysot.core.config import cfg
from Track.pysot.pysot.models.model_builder import ModelBuilder
from Track.pysot.pysot.tracker.tracker_builder import build_tracker
import dlib
import numpy as np
torch.set_num_threads(1)
parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, default="./Track/model-weights/track_config.yaml", help='config file')
parser.add_argument('--snapshot', type=str, default='./Track/model-weights/track_model.pth', help='model name')
args = parser.parse_args()

class Face_Detect_Track:
    def __init__(self):

        self.Box_frame = 0
        self.detect_box = True
        # self.face_detector = yoloFace()
        self.face_detector_quick = dlib.get_frontal_face_detector()
        self.tracker = self.track_define()

    def track_define(self):
        # load config
        cfg.merge_from_file(args.config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        # create model
        model = ModelBuilder()
        # load model
        model.load_state_dict(
            torch.load('./Track/model-weights/track_model.pth',
                       map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)
        # build tracker
        tracker = build_tracker(model)
        return tracker

    def detect(self, frame):
        if self.detect_box == True:
            boxes = self.face_detector_quick(frame)
            print("No. of faces Detected : {}".format(len(boxes)))
            if len(boxes) >= 1:
                boxes = [boxes[0].left(), boxes[0].top(), boxes[0].right() - boxes[0].left(), boxes[0].bottom() - boxes[0].top()]
                self.tracker.init(frame, boxes)
                # for x, y, w, h in boxes:
                #     im = frame[y:y + h, x:x + w]
                cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[0] + boxes[2], boxes[1]+boxes[3]), (255,0,255), 10)
                #cv2.imshow("part", im)
                self.Box_frame += 1
                self.detect_box = False
                return [boxes[1], boxes[1]+boxes[3], boxes[0], boxes[0] + boxes[2]]
            else:
                return [0, 0, 0, 0]
        else:
            outputs = self.tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            left_x = max(bbox[0], 0)
            left_y = max(bbox[1], 0)
            right_x = min(bbox[2], 640)
            right_y = min(bbox[3], 480)
            # im = frame[left_y:right_y, left_x:right_x]
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 10)
            # cv2.imshow("part", im)
            self.Box_frame += 1
            if self.Box_frame % 20 == 0 and self.detect_box == False:
                self.detect_box = True
            return [left_y, right_y, left_x, right_x]



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # 输入 frame
    # 输出  left_y:right_y, left_x:right_x
    detector = Face_Detect_Track()
    while True:
        ret, frame = cap.read()
        if ret:
            p = detector.detect(frame)
            # print(point)
            # for det in point:
            #     x = det.left()
            #     y = det.top()
            #     w = det.right()
            #     h = det.bottom()
            #     print(x, y, w, h)
            cv2.rectangle(frame, (p[2], p[0]), (p[3], p[1]), (255, 0, 0))
            # print(p)
        cv2.imshow('1', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
