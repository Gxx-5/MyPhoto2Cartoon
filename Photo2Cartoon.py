import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
from utils import Preprocess
import time


class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)

        assert os.path.exists(
            './models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load('./models/photo2cartoon_weights.pt', map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        start = time.time()
        face_rgba = self.pre.process_optimize(img)
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/face_rgba.png", face_rgba)
        end = time.time()
        print("preprocess time spent: ", end - start)

        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None, None, None

        print('[Step2: face detect] success!')
        size = [face_rgba.shape[0], face_rgba.shape[1]]
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        # mask = np.ones((256, 256, 1)) # cancel front background segmentation module
        face = (face * mask + (1 - mask) * 255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')

        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/cartoon_before.png", cartoon)
        # cv2.imwrite("/home/onepiece/GZX/photo2cartoon/photo2cartoon-master/images/mask_before.png", mask * 255)
        # resize back to origin size
        cartoon = cv2.resize(cartoon, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask * 1.1, (size[0], size[1]), interpolation=cv2.INTER_AREA)[:, :, np.newaxis].astype(
            np.uint8)

        return cartoon, mask, self.pre.rect
