import cv2
import numpy as np
from scipy.signal import convolve2d
import math
from PyQt5.QtGui import QImage, QPixmap

class ROI:
    # 建構式
    def __init__(self, viewer, rubberBand):
        self.img = None
        self.img_roi = None
        self.viewer = viewer
        self.rubberBand = rubberBand
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

    # 方法(Method)
    def set_x1_y1(self, x, y):
        self.x1 = x
        self.y1 = y

    def set_x2_y2(self, x, y):
        self.x2 = x
        self.y2 = y
    
    def set_img(self, img = None, viewer = None):
        if isinstance(img, np.ndarray):
            self.img = img
            self.viewer = viewer

        if isinstance(self.img, np.ndarray):
            qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*self.img.shape[2], QImage.Format_RGB888).rgbSwapped()
            self.viewer.setPhoto(QPixmap(qimg))

    def get_sharpness(self):
        I = self.img_roi
        return np.round(cv2.Laplacian(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()/100, 4)
    
    def get_noise(self):  
        I = self.img_roi
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        H, W = I.shape

        M = [[1, -2, 1],
           [-2, 4, -2],
           [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

        return np.round(sigma, 4)
    
    def get_ROI(self):
        if self.x2-self.x1 < 1 or self.y2-self.y1 < 1: return None
        # print(self.x1, self.y1, self.x2,self.y2)
        
        self.img_roi = self.img[self.y1:self.y2, self.x1:self.x2]
        return self.img_roi
