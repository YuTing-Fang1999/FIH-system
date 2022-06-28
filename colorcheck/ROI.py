import cv2
import numpy as np
from scipy.signal import convolve2d
import math
from PyQt5.QtGui import QImage, QPixmap

class ROI:
    # 建構式
    def __init__(self, viewer, rubberBand):
        self.y_start = 0
        self.x_start = 0
        self.y_end = 0
        self.x_end = 0
        self.img = None
        self.img_roi = None
        self.viewer = viewer
        self.rubberBand = rubberBand
        
    # 方法(Method)
    def set_x1_y1(self, x, y):
        self.x1 = x
        self.y1 = y

    def set_x2_y2(self, x, y):
        self.x2 = x
        self.y2 = y
    
    def set_img(self, img = None, viewer = None, default_ROI = None):
        if isinstance(img, np.ndarray):
            self.img = img
            self.viewer = viewer

        if isinstance(self.img, np.ndarray):
            qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*self.img.shape[2], QImage.Format_RGB888).rgbSwapped()
            self.viewer.setPhoto(QPixmap(qimg))

            # if isinstance(default_ROI, list):
            #     x1, y1, x2, y2 = default_ROI
            #     self.rubberBand.setGeometry(QtCore.QRect(QtCore.QPoint(x1,y1), QtCore.QPoint(x2,y2))) 
            #     self.rubberBand.show()

            #     self.set_x1_y1(x1,y1)
            #     self.set_x2_y2(x2,y2)

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
        viewer_w = self.viewer.width()
        viewer_h = self.viewer.height()

        self.img_roi = None
        if self.img is None: return None

        self.w = abs(self.x1-self.x2)
        self.h = abs(self.y1-self.y2)
#       print(self.w, self.h)
        if self.w==0 or self.h==0 : return None
        
        if(self.x1 > self.x2):
            self.x1 = self.x2
            self.y1 = self.y2
            
        if(self.img.shape[0]/self.img.shape[1] < viewer_h/viewer_w):
            y_start = int(self.y1 - viewer_h/2)
            x_start = self.x1
            x_end = x_start + self.w
            y_end = y_start + self.h
            
            factor = self.img.shape[1]/viewer_w
            x_start = int(x_start*factor)
            y_start = int(y_start*factor)
            x_end = int(x_end*factor)
            y_end = int(y_end*factor)
            
            y_start = int(y_start + self.img.shape[0]/2)
            y_end = int(y_end + self.img.shape[0]/2)
        else:
            x_start = int(self.x1 - viewer_w/2)
            y_start = self.y1
            x_end = x_start + self.w
            y_end = y_start + self.h
            
            factor = self.img.shape[0]/viewer_h
            x_start = int(x_start*factor)
            y_start = int(y_start*factor)
            x_end = int(x_end*factor)
            y_end = int(y_end*factor)
            
            x_start = int(x_start + self.img.shape[1]/2)
            x_end = int(x_end + self.img.shape[1]/2)
            
        if y_start<0 : y_start=0
        if x_start<0 : x_start=0
        if y_end<0 : y_end=0
        if x_end<0 : x_end=0
        
        if y_end>self.img.shape[0] : y_end=self.img.shape[0]
        if x_end>self.img.shape[1] : x_end=self.img.shape[0]
        if y_start>self.img.shape[0] : y_start=self.img.shape[0]
        if x_start>self.img.shape[1] : x_start=self.img.shape[0]

        if y_start-y_end==0 or x_start-x_end==0 : return None
        self.y_start = y_start
        self.x_start = x_start
        self.y_end = y_end
        self.x_end = x_end
        
        self.img_roi = self.img[y_start:y_end, x_start:x_end]
        return self.img_roi
