# import pandas as pd
import cv2

img1 = cv2.imread("dead-leaves-crop.jpg")

size = 3
img1 = cv2.GaussianBlur(img1,(size,size),0)

cv2.imwrite("dead-leaves-crop-blur.jpg", img1)