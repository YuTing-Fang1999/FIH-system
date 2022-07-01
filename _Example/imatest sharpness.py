import cv2 
import numpy as np

img = cv2.imdecode( np.fromfile( file = 'Dead-leaves-test-target.jpg', dtype = np.uint8 ), cv2.IMREAD_COLOR )

sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # x方向梯度 ksize默認為3x3
sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # y方向梯度

abx = np.abs(sobelx)  # 取sobelx絕對值
aby = np.abs(sobely)  # 取sobely絕對值

sx = np.mean(abx)/np.mean(img)
sy = np.mean(aby)/np.mean(img)

print(np.mean(abx))
print(np.mean(aby))
print(np.mean(img))
print(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F))