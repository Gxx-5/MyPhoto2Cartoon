import time
import cv2
import numpy as np

img = cv2.imread('D:\\shi\\a.png');
rows = img.shape[0]
cols = img.shape[1]
channels = img.shape[2]
mask = np.zeros(img.shape, dtype=np.uint8)
# 输入点的坐标
roi_corners = np.array([[(10, 10), (40, 20), (70, 80), (5, 100)]], dtype=np.int32)
channel_count = channels
ignore_mask_color = (255,) * channel_count
# 创建mask层
cv2.fillPoly(mask, roi_corners, ignore_mask_color)
# 为每个像素进行与操作，除mask区域外，全为0
masked_image = cv2.bitwise_and(img, mask)
cv2.imshow("src", masked_image)
cv2.waitKey(0)


def total_time(func):  # func = hell_word
    def wrapper():  # 等价于hell_word()
        start_time = time.time()
        func()
        end_time = time.time()
        print(end_time - start_time)  # 打印统计时间

    return wrapper


mask = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
src = [[12, 15, 16], [77, 88, 99], [45, 62, 82]]
tar = [[156, 0, 0], [154, 102, 0], [134, 121, 145]]
