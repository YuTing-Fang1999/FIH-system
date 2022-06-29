import cv2
import numpy as np
from scipy.signal import convolve2d
import math
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore

class ROI:
    # 建構式
    def __init__(self, viewer, rubberBand):
        self.viewer = viewer
        self.rubberBand = rubberBand

        self.img = None
        self.roi_img = None
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

        #colorcheck
        self.square_rate = 0.08
        self.padding_rate = 0.088
        self.start_h_rate = 0.038
        self.start_w_rate = 0.038

    def resetROI(self):
        self.rubberBand.hide()
        self.roi_img = None
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

        #colorcheck
        self.square_rate = 0.08
        self.padding_rate = 0.088
        self.start_h_rate = 0.038
        self.start_w_rate = 0.038

    def get_rubberBand_pos(self):
        return self.originPos, self.endPos

    # 方法(Method)
    def get_x1_y1_x2_y2(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def set_x1_y1_x2_y2(self, coor):
        self.x1 = coor[0]
        self.y1 = coor[1]
        self.x2 = coor[2]
        self.y2 = coor[3]


    def set_x1_y1(self, x, y):
        self.x1 = x
        self.y1 = y

    def set_x2_y2(self, x, y):
        self.x2 = x
        self.y2 = y
    
    def set_img(self, img = None, viewer = None):
        self.resetROI()
        if isinstance(img, np.ndarray):
            self.img = img
            self.viewer = viewer

        if isinstance(self.img, np.ndarray):
            qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*self.img.shape[2], QImage.Format_RGB888).rgbSwapped()
            self.viewer.setPhoto(QPixmap(qimg))
            self.viewer.setFocus()

    def get_sharpness(self):
        I = self.roi_img
        return np.round(cv2.Laplacian(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()/100, 4)
    
    def get_noise(self):  
        I = self.roi_img
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        H, W = I.shape

        M = [[1, -2, 1],
           [-2, 4, -2],
           [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

        return np.round(sigma, 4)
    
    def set_ROI(self):
        if self.x2-self.x1 < 1 or self.y2-self.y1 < 1: return None
        self.roi_img = self.img[self.y1:self.y2, self.x1:self.x2]

        color = (0, 0, 255) # red
        thickness = 5 # 寬度 (-1 表示填滿)
        img = self.img.copy()
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)

        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*img.shape[2], QImage.Format_RGB888).rgbSwapped()
        self.viewer.setPhoto(QPixmap(qimg))

    def get_roi_img_by_roi_coordinate(self):
        roi = []
        for coor in self.roi_coordinate:
            roi.append(self.roi_img[int(coor[1]):int(coor[3]), int(coor[0]):int(coor[2])])

        return roi

    def set_24_block_roi_coordinate(self):
        coordinate = []
        h, w, c = self.roi_img.shape
        square = self.square_rate*w
        padding = self.padding_rate*w

        start_h = self.start_h_rate*w
        for i in range(4):
            start_w = self.start_w_rate*w
            for j in range(6):
                coordinate.append([int(start_w), int(start_h), int(start_w+square), int(start_h+square)])
                start_w+=(square+padding)
            start_h+=(square+padding)

        self.roi_coordinate = coordinate

    def get_rectangle_img_by_roi_coordinate(self):
        color = (0, 0, 255) # red
        thickness = self.roi_img.shape[1]//200 # 寬度 (-1 表示填滿)

        img = self.roi_img.copy()
        for i in range(24):
            cv2.rectangle(img, (self.roi_coordinate[i][0], self.roi_coordinate[i][1]), (self.roi_coordinate[i][2], self.roi_coordinate[i][3]), color, thickness)
        return self.ResizeWithAspectRatio(img, width = 400)

    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def get_resize_roi_img(self, width = 400):
        return self.ResizeWithAspectRatio(self.roi_img, width)
