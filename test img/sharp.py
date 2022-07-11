import cv2
import numpy as np

img = cv2.imread("grid.jpg")

I = img.copy()
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
gamma=0.5
invGamma = 1.0 / gamma
I = np.array(((I / 255.0) ** invGamma) * 255) # linearized
g = cv2.Laplacian(I, cv2.CV_64F, ksize=1)

v = np.round(np.sqrt(g.var()), 4)

print(v)


sobelx = cv2.Sobel(I, cv2.CV_16S, 1, 0, ksize=3)  # x方向梯度 ksize默認為3x3
sobely = cv2.Sobel(I, cv2.CV_16S, 0, 1, ksize=3)  # y方向梯度

abx = np.abs(sobelx)  # 取sobelx絕對值
aby = np.abs(sobely)  # 取sobely絕對值
print(np.sqrt(sobelx.var()))
print(np.mean(I))