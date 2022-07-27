# import pandas as pd
import cv2
import numpy as np
from scipy.signal import convolve2d

def get_gamma(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
    gamma = 0.5
    invGamma = 1.0 / gamma
    I = np.array(((I / 255.0) ** invGamma) * 255)  # linearized
    return I

img1 = cv2.imread("test1.jpg")
img2 = cv2.imread("test2.jpg")
img3 = cv2.imread("test3.jpg")
img4 = cv2.imread("test4.jpg")

# remove noise
# size = 3
# img1 = cv2.GaussianBlur(img1,(size,size),0)
# img2 = cv2.GaussianBlur(img2,(size,size),0)
# img3 = cv2.GaussianBlur(img3,(size,size),0)
# img4 = cv2.GaussianBlur(img4,(size,size),0)

img1 = get_gamma(img1.copy())
img2 = get_gamma(img2.copy())
img3 = get_gamma(img3.copy())
img4 = get_gamma(img4.copy())

# cv2.imshow("img", img1/255)
# cv2.imshow("img2", img4/255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

n = 2
kernal_x = np.array([
    [-1, 0, 1],
])/n

kernal_y = np.array([
    [-1],
    [0],
    [1],
])/n

H = cv2.filter2D(img4, cv2.CV_64F, kernal_x)
V = cv2.filter2D(img4, cv2.CV_64F, kernal_y)
# print(H.std(), V.std())

gy, gx = np.gradient(img4)
# print(gx.std(), gy.std())
# print(np.mean(np.abs(gx))/np.mean(img4)) # 官網公式
# print(np.mean(gx**2)/np.mean(img4))
# print(np.mean(gx**2))

kernal_x = np.array([
    [1, -1],
])/2

def sobelx(I):
    return cv2.Sobel(I,cv2.CV_64F,1,0,ksize=3)

def sobely(I):
    return cv2.Sobel(I,cv2.CV_64F,0,1,ksize=3)

def scharrx(I):
    return cv2.Scharr(I,cv2.CV_64F,1,0)

def g1x(I):
    gy, gx = np.gradient(I)
    return gx

def g2x(I):
    return cv2.filter2D(I, cv2.CV_64F, np.array([[-1, 1]]))

def g2y(I):
    return cv2.filter2D(I, cv2.CV_64F, np.array([[-1], [1]]))

func = sobely
print(np.mean(np.abs(func(img1)))/np.mean(img1)) # 官網公式
print(np.mean(np.abs(func(img2)))/np.mean(img2)) # 官網公式
# print(np.mean(np.abs(func(img3)))/np.mean(img3)) # 官網公式
print(np.mean(np.abs(func(img4)))/np.mean(img4)) # 官網公式

# print(func(img1).std()/np.mean(img1))
# print(func(img2).std()/np.mean(img2))
# print(func(img4).std()/np.mean(img4))

# n = 8
# # print(n)
# kernal_x = np.array([
#     [-1,  0,  1],
#     [-2,  0,  2, ],
#     [-1,  0,  1, ]
# ])/n

# kernal_y = np.array([
#     [-1, -2, -1],
#     [0,  0,  0, ],
#     [1,  2,  1, ]
# ])/n

# scharrx=cv2.Scharr(I,cv2.CV_64F,1,0)
# scharry=cv2.Scharr(I,cv2.CV_64F,0,1)
# scharrx=cv2.convertScaleAbs(scharrx)
# scharry=cv2.convertScaleAbs(scharry)

# ksize = 1
# sobelx1=cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=ksize)
# sobely1=cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=ksize)

# sobelx4=cv2.Sobel(img4,cv2.CV_64F,1,0,ksize=ksize)
# sobely4=cv2.Sobel(img4,cv2.CV_64F,0,1,ksize=ksize)

# sobely=cv2.Sobel(I,cv2.CV_64F,0,2,ksize=1)
# sobelx1=cv2.convertScaleAbs(sobelx1)
# sobelx4=cv2.convertScaleAbs(sobelx4)

# sobelx1 = sobelx1[sobelx1!=0]
# sobelx4 = sobelx4[sobelx4!=0]

# print(sobelx1.std())
# print(sobely1.std())
# print(pd.DataFrame(sobelx1.reshape(-1)).describe())

# print(sobelx4.std())
# print(sobely4.std())
# print(pd.DataFrame(sobelx4.reshape(-1)).describe())

# gy, gx = np.gradient(img1)
# print(gx.std(), gy.std())
# print(gx[gx!=0].std())
# print(gy[gy!=0].std())


# gy, gx = np.gradient(img4)
# print(gx.std(), gy.std())
# print(gx[gx!=0].std())
# print(gy[gy!=0].std())

# H1 = cv2.filter2D(img1, cv2.CV_64F, kernal_x)
# H4 = cv2.filter2D(img4, cv2.CV_64F, kernal_x)
# H1 = cv2.convertScaleAbs(H1)
# H4 = cv2.convertScaleAbs(H4)
# H1 = H1[H1 != 0]
# H4 = H4[H4 != 0]

# df_describe = pd.DataFrame(sobelx.reshape(-1))
# print(df_describe.describe())


