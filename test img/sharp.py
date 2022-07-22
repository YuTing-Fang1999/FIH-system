import pandas as pd
import cv2
import numpy as np
from scipy.signal import convolve2d

def gamma(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    gamma = 0.5
    invGamma = 1.0 / gamma
    I = np.array(((I / 255.0) ** invGamma) * 255)  # linearized
    return I.astype('int8')

img1 = cv2.imread("test1.jpg")
img4 = cv2.imread("test4.jpg")

# remove noise
size = 3
img1 = cv2.GaussianBlur(img1,(size,size),0)
img4 = cv2.GaussianBlur(img4,(size,size),0)

img1 = gamma(img1.copy())
img4 = gamma(img4.copy())

# cv2.imshow("img", img1)
# cv2.imshow("img2", img4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img1 = img1.astype('float64')
img4 = img4.astype('float64')

n = 2
# print(n)
kernal_x = np.array([
    [-1, 0, 1],
])/n

kernal_y = np.array([
    [-1],
    [0],
    [1],
])/n

# H = cv2.filter2D(I, cv2.CV_64F, kernal_x)
# V = cv2.filter2D(I, cv2.CV_64F, kernal_y)
# print(H.std())
# print(V.std())
# print(np.sqrt((H.var() + V.var())/2))


n = 8
# print(n)
kernal_x = np.array([
    [-1,  0,  1],
    [-2,  0,  2, ],
    [-1,  0,  1, ]
])/n

kernal_y = np.array([
    [-1, -2, -1],
    [0,  0,  0, ],
    [1,  2,  1, ]
])/n

# scharrx=cv2.Scharr(I,cv2.CV_64F,1,0)
# scharry=cv2.Scharr(I,cv2.CV_64F,0,1)
# scharrx=cv2.convertScaleAbs(scharrx)
# scharry=cv2.convertScaleAbs(scharry)

ksize = 1
sobelx1=cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=ksize)
sobely1=cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=ksize)

sobelx4=cv2.Sobel(img4,cv2.CV_64F,1,0,ksize=ksize)
sobely4=cv2.Sobel(img4,cv2.CV_64F,0,1,ksize=ksize)

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

gy, gx = np.gradient(img1)
print(gx.std(), gy.std())
# print(gx[gx!=0].std())
# print(gy[gy!=0].std())


gy, gx = np.gradient(img4)
print(gx.std(), gy.std())
# print(gx[gx!=0].std())
# print(gy[gy!=0].std())

# H1 = cv2.filter2D(img1, cv2.CV_64F, kernal_x)
# H4 = cv2.filter2D(img4, cv2.CV_64F, kernal_x)
# H1 = cv2.convertScaleAbs(H1)
# H4 = cv2.convertScaleAbs(H4)
# H1 = H1[H1 != 0]
# H4 = H4[H4 != 0]

# print(pd.DataFrame(H1.reshape(-1)).describe())
# print(pd.DataFrame(H4.reshape(-1)).describe())
# print(H1.std(), H4.std())
# H = cv2.filter2D(H, cv2.CV_64F, kernal_x)
# V = cv2.filter2D(I, cv2.CV_64F, kernal_y)
# V = cv2.filter2D(V, cv2.CV_64F, kernal_y)

# df_describe = pd.DataFrame(sobelx.reshape(-1))
# print(df_describe.describe())



# H = H[np.abs(H) != 0]
# V = V[np.abs(V) != 0]
# print(H.std())
# print(V.std())

