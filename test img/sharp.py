import cv2
import numpy as np
from scipy.signal import convolve2d

img = cv2.imread("test_grid.jpg")

I = img.copy()
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('float64')
gamma = 0.5
invGamma = 1.0 / gamma
I = np.array(((I / 255.0) ** invGamma) * 255)  # linearized

# I = np.ones([4,4])
# I[1:3,1:3] = 0
# print(I)


g = cv2.Laplacian(I, cv2.CV_64F, ksize=1)

v = np.round(g.var(), 4)

print(v)
print()


sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=1)  # x方向梯度 ksize默認為3x3
sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=1)  # y方向梯度

abx = np.abs(sobelx)  # 取sobelx絕對值
aby = np.abs(sobely)  # 取sobely絕對值
# print(sobelx.var()+sobely.var())
# print(sobely.var())
# print()

# print(sobelx.var())
# print(np.mean(sobelx**2))


# # 拉普拉斯濾波器
# laplace_filter = np.array([
#     [0, 1, 0],
#     [1, -4, 1],
#     [0, 1, 0],
# ])

# laplace_filter = np.array([
#     [-1, -2, -1],
#     [ 0,  0,  0,],
#     [ 1,  2,  1,]
# ])

# laplace_filter = np.array([
#     [-1],
#     [ 0],
#     [ 1]
# ])

laplace_filter = np.array([
    [-1, -2, -1],
    [0,  0,  0, ],
    [1,  2,  1, ]
])


g = cv2.filter2D(I, cv2.CV_64F, laplace_filter)
# v = np.round(np.sqrt(g.var()), 4)
# print(g)
# print(np.abs(g).mean())
print(g.var())

# sobel5x = cv2.getDerivKernels(2, 0, 3)
# print(np.outer(sobel5x[0], sobel5x[1]))

img = np.zeros([1000, 1000])

# for i in range(0,10,2):
#     img[i*100:(i+1)*100] = 255

img[500:, :] = 255

mean = 0
sigma = 0.1
# int -> float (標準化)
img = img / 255
# 隨機生成高斯 noise (float + float)
noise = np.random.normal(mean, sigma, img.shape)
# noise + 原圖
gaussian_out = img + noise
# 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
gaussian_out = np.clip(gaussian_out, 0, 1)

# 原圖: float -> int (0~1 -> 0~255)
gaussian_out = np.uint8(gaussian_out*255)

cv2.imshow('img', gaussian_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('test_grid2.jpg', gaussian_out)
