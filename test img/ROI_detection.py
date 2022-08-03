import random
import cv2
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

im=cv2.imread('DXO accurate/A_5.jpg')
# im=cv2.imread('dead-leaves.jpg')
im = ResizeWithAspectRatio(im, height=1000)

segmentator=cv2.ximgproc.segmentation.createGraphSegmentation(
    sigma = 1,
    k=1000,
    min_size =1000
)
im = cv2.medianBlur(im, 3)

segment = segmentator.processImage(im)
segment = np.uint8(segment)
segment = cv2.medianBlur(segment, 3)
# cv2.imshow("seg", ((segment/np.max(segment))*255).astype('uint8'))
# cv2.waitKey(0)

his, bins = np.histogram(segment, bins = np.max(segment))
print(his)
print(np.argsort(his))

for i in np.argsort(his)[-2:-1]:
  y, x = np.where(segment == i)

  # 計算每分割區域的上下左右邊界
  top, bottom, left, right = min(y), max(y), min(x), max(x)

  # 隨機產生顏色
  color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

  # 繪製方框
  cv2.rectangle(im, (left, bottom), (right, top), color, 5)
  cv2.imshow("Result", ResizeWithAspectRatio(im, height=900))
  cv2.waitKey(0)

# 將原始圖片與分割區顏色合併
# result = cv2.addWeighted(im, 0.3, seg_image, 0.7, 0)
# cv2.imshow("Result", result)

# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = ResizeWithAspectRatio(gray, height=600)
# edged = cv2.Canny(gray, 170, 490)
# cv2.imshow("im",edged)

# cv2.waitKey(0)
# cv2.destroyAllWindows()