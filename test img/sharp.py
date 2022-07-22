import pandas as pd
import cv2
import numpy as np
from scipy.signal import convolve2d

def gamma(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
    gamma = 0.5
    invGamma = 1.0 / gamma
    I = np.array(((I / 255.0) ** invGamma) * 255)  # linearized
    return I

img1 = cv2.imread("test1.jpg")
img4 = cv2.imread("test4.jpg")

img1 = gamma(img1.copy())
img4 = gamma(img4.copy())

# I = cv2.blur(I, (3,3))

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

sobelx=cv2.Sobel(I,cv2.CV_64F,2,0,ksize=1)
sobely=cv2.Sobel(I,cv2.CV_64F,0,2,ksize=1)
sobelx=cv2.convertScaleAbs(sobelx)
sobely=cv2.convertScaleAbs(sobely)

print(sobelx.std())
print(sobely.std())

# H = cv2.filter2D(sobelx, cv2.CV_64F, kernal_x)
# H = cv2.filter2D(H, cv2.CV_64F, kernal_x)
# V = cv2.filter2D(I, cv2.CV_64F, kernal_y)
# V = cv2.filter2D(V, cv2.CV_64F, kernal_y)

# H=cv2.convertScaleAbs(H)
# V=cv2.convertScaleAbs(V)



df_describe = pd.DataFrame(sobelx.reshape(-1))
print(df_describe.describe())

print(cv2.Laplacian(I, cv2.CV_64F, ksize=1).std())

# H = H[np.abs(H) != 0]
# V = V[np.abs(V) != 0]
# print(H.std())
# print(V.std())

# print(scharrx.std())
# print(scharry.std())
