from tkinter.messagebox import NO
import cv2
import numpy as np
from scipy.signal import convolve2d
import math

class ROI:
    # 建構式
    def __init__(self):
        self.img = None
        self.roi_img = None
        self.roi_coordinate = None

    def set_roi_img(self, img, roi_coordinate):
        self.img = img
        self.roi_coordinate = roi_coordinate
        coor = self.roi_coordinate
        self.roi_img = self.img[int(coor.r1):int(coor.r2), int(coor.c1):int(coor.c2), :]
        

    ###### sharpness ######
    def get_sharpness(self, img):
        I = img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        return np.round(cv2.Laplacian(I, cv2.CV_64F).var()/100, 4)
    
    def get_noise(self, img):  
        I = img.copy()
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        H, W = I.shape

        M = [[1, -2, 1],
           [-2, 4, -2],
           [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

        return np.round(sigma, 4)
    
    
    ###### colorcheck ######
    def gen_colorcheck_coordinate(self):
        coor = []
        h, w, c = self.get_roi_img_by_roi_coordinate().shape
        square = self.square_rate*w
        padding = self.padding_rate*w

        start_h = self.start_h_rate*w
        for i in range(4):
            start_w = self.start_w_rate*w
            for j in range(6):
                coor.append([int(start_w), int(start_h), int(start_w+square), int(start_h+square)])
                start_w+=(square+padding)
            start_h+=(square+padding)

        return coor

    def set_colorcheck_coordinate(self, coor):
        self.colorcheck_coordinate = coor

    def get_colorcheck_roi_draw(self):
        color = (0, 0, 255) # red
        thickness = self.roi_img.shape[1]//200 # 寬度 (-1 表示填滿)

        img = self.roi_img.copy()
        for i in range(24):
            cv2.rectangle(img, (self.roi_coordinate[i][0], self.roi_coordinate[i][1]), (self.roi_coordinate[i][2], self.roi_coordinate[i][3]), color, thickness)
        return img

    
