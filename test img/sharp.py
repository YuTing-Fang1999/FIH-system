import pandas as pd
import cv2
import numpy as np
from scipy.signal import convolve2d

img = cv2.imread("test4.jpg")

I = img.copy()
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
gamma = 0.5
invGamma = 1.0 / gamma
I = np.array(((I / 255.0) ** invGamma) * 255)  # linearized

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

H = cv2.filter2D(I, cv2.CV_64F, kernal_x)
V = cv2.filter2D(I, cv2.CV_64F, kernal_y)
# print(H.std())
# print(V.std())
# print(np.sqrt((H.var() + V.var())/2))


# n = 8
# print(n)
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

H = cv2.filter2D(I, cv2.CV_64F, kernal_x)
V = cv2.filter2D(I, cv2.CV_64F, kernal_y)

df_describe = pd.DataFrame(H.reshape(-1))
print(df_describe.describe())

# H = H[np.abs(H) > 0.1]
# V = V[np.abs(V) > 0.1]
print(H.std())
print(V.std())
# print(np.sqrt((H.var() + V.var())/2))
# H = H[H > 0.1]
