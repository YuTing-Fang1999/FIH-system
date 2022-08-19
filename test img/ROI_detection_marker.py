import random
import cv2
import numpy as np

def get_gamma(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
    gamma = 0.5
    invGamma = 1.0 / gamma
    I = np.array(((I / 255.0) ** invGamma) * 255)  # linearized
    return I

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

# im=cv2.imread('OPPO Find X2 DLC/A_5.jpg')
# im=cv2.imread('OPPO Find X2 DLC/A_20.jpg')
# im=cv2.imread('OPPO Find X2 DLC/A_100.jpg')
# im=cv2.imread('OPPO Find X2 DLC/A_300.jpg')
im=cv2.imread('OPPO Find X2 DLC/D65_1000.jpg')
# im=cv2.imread('OPPO Find X2 DLC/H_1.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_20.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_100.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_300.jpg')
# im=cv2.imread('OPPO Find X2 DLC/TL84_1000.jpg')

# im=cv2.imread('dead-leaves.jpg')
# resize_im = ResizeWithAspectRatio(im, height=800)
resize_im = im
gray = cv2.cvtColor(resize_im, cv2.COLOR_BGR2GRAY)


# remove noise
# size = 3
# gray = cv2.GaussianBlur(gray,(size,size),0)
# cv2.imshow("cammy",gray)
# cv2.waitKey(0)




edged = cv2.Canny(gray, 300, 600)
kernel = np.ones((2,2), np.uint8) 
# edged = cv2.dilate(edged, kernel, iterations = 1)
cv2.imshow("cammy",edged)
cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
clone = resize_im.copy()

coor = []
# 依次處理每個Contours

find = False

for i, c in enumerate(cnts):

    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: continue
    solidity = float(area)/hull_area

    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h

    if np.around(solidity, 1) == 0.6 and np.around(aspect_ratio, 1) == 1:

        (c,r),(MA,ma),angle = cv2.fitEllipse(c)
        # 往右下遞增
        r, c = int(r), int(c)

        # 避免重複尋找
        if find and np.linalg.norm(np.array(coor[-1])-np.array([r,c]))<5:
            continue

        if not find:
            find = True

        coor.append((r,c)) # row, col
        # 隨機產生顏色
        color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]
        cv2.drawContours(clone, cnts, i, color, -1)
        # 在中心點畫上黃色實心圓
        cv2.circle(clone, (c, r), 2, (1, 227, 254), -1)
        cv2.putText(clone, "({}, {})".format(r, c), (c-30, r), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        # show img
        cv2.imshow("contours",clone)
        cv2.waitKey(0)

print(len(coor))
# assert len(coor) == 4

# coor = np.array(coor)
# scale = im.shape[0]/resize_im.shape[0]
# coor = np.around(coor * scale).astype(int)

# # 由下到上，右到左
# for c in coor:
#     # 在中心點畫上黃色實心圓
#     cv2.circle(im, (c[1], c[0]), 10, (1, 227, 254), -1)
#     cv2.putText(im, "({}, {})".format(c[0], c[1]), (c[1]-30, c[0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10, cv2.LINE_AA)

# # find center
# # 往右下的方向
# vec = coor[0] - coor[3] # → ↓
# mid = coor[3] + vec/2
# mid = np.around(mid).astype(int)
# cv2.circle(im, (mid[1], mid[0]), 20, (1, 227, 254), -1)
# # print(mid)

# # find target ROI
# rate = np.linalg.norm(vec)*0.2
# vec = np.array([1,1])*rate
# topLeft = np.around(mid - vec).astype(int)
# bottomRight = np.around(mid + vec).astype(int)
# # 繪製方框
# cv2.rectangle(im, (topLeft[1], topLeft[0]), (bottomRight[1], bottomRight[0]), (255,0,0), 10)

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

cv2.imshow("contours", ResizeWithAspectRatio(im, height=600))
cv2.waitKey(0)
cv2.destroyAllWindows()


