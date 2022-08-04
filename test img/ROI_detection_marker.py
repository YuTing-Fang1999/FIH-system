import random
import cv2
import numpy as np
from sympy import bottom_up

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

# im=cv2.imread('DXO accurate/A_5_1.226.jpg')
im=cv2.imread('dead-leaves.jpg')
im = ResizeWithAspectRatio(im, height=500)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 300, 600)
kernel = np.ones((2,1), np.uint8) 
edged = cv2.dilate(edged, kernel, iterations = 1)

# cv2.imshow("cammy",edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

clone = im.copy()


coor = []
# 依次處理每個Contours

for i, c in enumerate(cnts):

    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: continue
    solidity = float(area)/hull_area

    if np.around(solidity, 1) == 0.6:
        # 隨機產生顏色
        color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]
        cv2.drawContours(clone, cnts, i, color, -1)

        (x,y),(MA,ma),angle = cv2.fitEllipse(c)
        x, y = int(x), int(y)
        # print(angle)
        coor.append((x,y))
        # 在中心點畫上黃色實心圓

        cv2.circle(clone, (x, y), 2, (1, 227, 254), -1)
        cv2.putText(clone, "({}, {})".format(x, y), (x-30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # cv2.imshow("contours",clone)
        # cv2.waitKey(0)

coor = np.array(coor)
vecX = coor[0] - coor[1]
vecY = coor[0] - coor[2]

midX = coor[1] + vecX/2
midY = coor[2] + vecY/2
mid = (midX[0], midY[1])
mid -= vecY*0.01
mid = mid.astype(int)
# print(mid)
w = np.linalg.norm(vecX)
h = np.linalg.norm(vecY)
normVecX = vecX / w
normVecY = vecY / h
vec = normVecX+normVecY

rate = w*0.25
vec*=rate
topLeft = (mid - vec).astype(int)
bottomRight = (mid + vec).astype(int)
# print(topLeft, bottomRight)
cv2.circle(clone, mid, 2, (1, 227, 254), -1)
# 繪製方框
cv2.rectangle(clone, topLeft, bottomRight, (255,0,0), 1)
# print(midX[0].dtype, h)
# test
# i = 50
# # 隨機產生顏色
# color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]
# cv2.drawContours(clone, cnts[i], -1, color, 2)

# x,y,w,h = cv2.boundingRect(cnts[i])
# aspect_ratio = float(w)/h

# area = cv2.contourArea(cnts[i])
# hull = cv2.convexHull(cnts[i])
# hull_area = cv2.contourArea(hull)
# solidity = float(area)/hull_area

# area = cv2.contourArea(cnts[i])
# equi_diameter = np.sqrt(4*area/np.pi)

# print(aspect_ratio, solidity, equi_diameter)

cv2.imshow("contours",clone)
cv2.waitKey(0)
cv2.destroyAllWindows()


