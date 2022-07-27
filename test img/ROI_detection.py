import random
import cv2
import numpy as np
im=cv2.imread('dead-leaves.jpg')
segmentator=cv2.ximgproc.segmentation.createGraphSegmentation(
    sigma = 0.5,
    k=300,
    min_size =100000
)
segment = segmentator.processImage(im)
seg_image = np.zeros(im.shape, np.uint8)

for i in range(np.max(segment)):
  y, x = np.where(segment != i)

  # 計算每分割區域的上下左右邊界
  top, bottom, left, right = min(y), max(y), min(x), max(x)

  # 繪製方框
  cv2.rectangle(im, (left, bottom), (right, top), (0, 255, 0), 1)

cv2.imshow("Result", im[top:bottom, left:right])
cv2.waitKey(0)
cv2.destroyAllWindows()