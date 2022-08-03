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

im=cv2.imread('DXO accurate/A_5_1.226.jpg')
im = ResizeWithAspectRatio(im, height=1000)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 400, 510)
cv2.imshow("cammy",edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

clone = im.copy()

cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)

# # 依次處理每個Contours

# for c in cnts:

#         # CV2.moments會傳回一系列的moments值，我們只要知道中點X, Y的取得方式是如下進行即可。

#         M = cv2.moments(c)

#         cX = int(M['m10'] / M['m00'])

#         cY = int(M['m01'] / M['m00'])

#         # 在中心點畫上黃色實心圓

#         cv2.circle(clone, (cX, cY), 10, (1, 227, 254), -1)

# cv2.imshow("contours",clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

# segmentator=cv2.ximgproc.segmentation.createGraphSegmentation(
#     sigma = 1,
#     k=1000,
#     min_size =1000
# )
# im = cv2.medianBlur(im, 3)

# segment = segmentator.processImage(im)
# segment = np.uint8(segment)
# segment = cv2.medianBlur(segment, 3)
# cv2.imshow("seg", ((segment/np.max(segment))*255).astype('uint8'))
# cv2.waitKey(0)

# his, bins = np.histogram(segment, bins = np.max(segment))
# print(his)
# print(np.argsort(his))

# for i in np.argsort(his)[-2:-1]:
#   y, x = np.where(segment == i)

#   # 計算每分割區域的上下左右邊界
#   top, bottom, left, right = min(y), max(y), min(x), max(x)

#   # 隨機產生顏色
#   color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

#   # 繪製方框
#   cv2.rectangle(im, (left, bottom), (right, top), color, 5)
#   cv2.imshow("Result", ResizeWithAspectRatio(im, height=900))
#   cv2.waitKey(0)
