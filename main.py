import cv2
from Photo2Cartoon import Photo2Cartoon
import time
import argparse
import numpy as np
import sys

c2p = Photo2Cartoon()


def PasteOnImg(img_src, img_tar, mask, rect):
    mask_extract = mask[rect[1][0]:rect[1][1] + 1, rect[1][2]:rect[1][3] + 1]
    tar_extract = img_tar[rect[1][0]:rect[1][1] + 1, rect[1][2]:rect[1][3] + 1]
    src_extract = img_src[rect[0][0]:rect[0][1] + 1, rect[0][2]:rect[0][3] + 1]

    mask_inv = np.subtract(np.ones((mask_extract.shape[0], mask_extract.shape[1], mask_extract.shape[2])), mask_extract)
    src_extract = np.multiply(mask_inv, src_extract) + tar_extract
    img_src[rect[0][0]:rect[0][1] + 1, rect[0][2]:rect[0][3] + 1] = src_extract.copy()
    return img_src


def Process(img):
    start = time.time()
    cartoon, mask, rect = c2p.inference(img)
    if cartoon is None:
        return
    end = time.time()
    print("Photo2Cartoon Time Spent: ", end - start)

    if args.source == 0:
        cv2.imwrite(args.save_path, cartoon)
        print('Cartoon portrait has been saved successfully!')
    elif args.source == 1 or 2:
        # cv2.imshow("cartoon", cartoon)
        cv2.imshow("img", img[:, :, [2, 1, 0]])
        cv2.imshow("effect", PasteOnImg(img, cartoon, mask, rect))
        cv2.waitKey(1)


if __name__ == '__main__':
    # photo_path = "./images/photo_test.jpg"
    # video_path = "./images/video_test.mp4"
    # save_path = "./images/cartoon_result.jpg"
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--source', type=int, default=0, help='input source 0-picture 1-camera 2-video')
    parser.add_argument('--photo_path', default="./images/photo_test.jpg", help='input image path if source is 0')
    parser.add_argument('--save_path', default="./images/cartoon_result.jpg", help='output image path if source is 0')
    parser.add_argument('--video_path', default="./images/video_test.mp4", help='input video path if source is 2')
    args = parser.parse_args()

    if args.source == 0:
        img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
        Process(img)
    else:
        if args.source == 1:
            cap = cv2.VideoCapture(0)
        elif args.source == 2:
            cap = cv2.VideoCapture(args.video_path)
        else:
            print("input wrong!")
            sys.exit()
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                continue
            # img = cv2.cvtColor(cv2.VideoCapture(0).read()[1], cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(cv2.VideoCapture(args.video_path), cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Process(img)
