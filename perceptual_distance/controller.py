
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from .UI import Ui_MainWindow
from myPackage.selectROI_window import SelectROI_window
from myPackage.ImageMeasurement import get_perceptual_distance, get_roi_img


import sys
import cv2
import numpy as np
sys.path.append("..")


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.selectROI_window = SelectROI_window()
        self.setup_control()
        self.tab_idx = 0

    def setup_control(self):
        # 須個別賦值(不能用for迴圈)，否則都會用到同一個數值
        self.ui.open_img_btn[0].clicked.connect(lambda: self.open_img(0))
        self.ui.open_img_btn[1].clicked.connect(lambda: self.open_img(1))
        self.ui.open_img_btn[2].clicked.connect(lambda: self.open_img(2))
        self.ui.open_img_btn[3].clicked.connect(lambda: self.open_img(3))
        self.selectROI_window.to_main_window_signal.connect(self.set_roi_coordinate)
        self.ui.btn_compute.clicked.connect(self.compute)

    def closeEvent(self, e):
        for i in range(4): 
            self.ui.img_block[i].hide()
            self.ui.score_region[i].hide()

    def open_img(self, tab_idx):
        self.selectROI_window.open_img(tab_idx)

    def set_roi_coordinate(self, img_idx, img, roi_coordinate, filename):
        roi_img = get_roi_img(img, roi_coordinate)
        # cv2.imshow("roi_img"+str(img_idx), roi_img)
        # cv2.waitKey(0)
        self.ui.img_block[img_idx].img = img
        self.ui.img_block[img_idx].roi_coordinate = roi_coordinate
        self.ui.img_block[img_idx].filename = filename
        # print(self.ui.img_block[img_idx].img.shape)
        # print(self.ui.img_block[img_idx].roi_coordinate.r1, self.ui.img_block[img_idx].roi_coordinate.r2)
        self.ui.img_block[img_idx].setPhoto(roi_img, filename)
        self.ui.filename[img_idx].setText(filename)
        self.ui.img_block[img_idx].show()
        # print(self.ui.img_block[img_idx].img.shape)
        # print(self.ui.img_block[img_idx].roi_coordinate.r1, self.ui.img_block[img_idx].roi_coordinate.r2)
        
        for i in range(4): 
            self.ui.score_region[i].hide()

    def compute(self, img_idx):
        if self.ui.img_block[0].img is None:
            QMessageBox.about(self, "info", "要先Load Ref Pic")
            return False
        
        # 得到Ref Pic的解析度
        h0, w0, c0 = self.ui.img_block[0].img.shape

        # 得到第一張照片的解析度
        i = 1
        while i<4:
            if self.ui.img_block[i].img is not None: break
            i+=1

        if i==4:
            QMessageBox.about(self, "info", "要Load Pic1")
            return False

        h1, w1, c1 = self.ui.img_block[i].img.shape
        # 確認Pic1~Pic3的解析度是否相同
        while i<4:
            if self.ui.img_block[i].img is not None: 
                h2, w2, c2 = self.ui.img_block[i].img.shape
                if h1!=h2 or w1!=w2:
                    QMessageBox.about(self, "info", "Pic1~Pic3的解析度(長寬大小)需相同")
                    return False
            i+=1

        # 解析度較大的那張照片要縮小
        if h0>h1 and w0>w1:
            # h0縮成h1後，w會縮減成h1/h0倍
            self.ui.img_block[0].img = cv2.resize(self.ui.img_block[0].img, (int(w0*(h1/h0)), h1), interpolation=cv2.INTER_AREA)
            self.ui.img_block[0].roi_coordinate.r1 = int(self.ui.img_block[0].roi_coordinate.r1 * h1/h0)
            self.ui.img_block[0].roi_coordinate.r2 = int(self.ui.img_block[0].roi_coordinate.r2 * h1/h0)
            self.ui.img_block[0].roi_coordinate.c1 = int(self.ui.img_block[0].roi_coordinate.c1 * h1/h0)
            self.ui.img_block[0].roi_coordinate.c2 = int(self.ui.img_block[0].roi_coordinate.c2 * h1/h0)
        elif h0<h1 and w0<w1:
            # h1縮成h0後，w會縮減成h0/h1倍
            for i in range(1, 4):
                if self.ui.img_block[i].img is not None:
                    print(self.ui.img_block[i].img.shape)
                    self.ui.img_block[i].img = cv2.resize(self.ui.img_block[i].img, (int(w1*(h0/h1)),h0), interpolation=cv2.INTER_AREA)
                    self.ui.img_block[i].roi_coordinate.r1 = int(self.ui.img_block[i].roi_coordinate.r1 * h0/h1)
                    self.ui.img_block[i].roi_coordinate.r2 = int(self.ui.img_block[i].roi_coordinate.r2 * h0/h1)
                    self.ui.img_block[i].roi_coordinate.c1 = int(self.ui.img_block[i].roi_coordinate.c1 * h0/h1)
                    self.ui.img_block[i].roi_coordinate.c2 = int(self.ui.img_block[i].roi_coordinate.c2 * h0/h1)
                    print(self.ui.img_block[i].img.shape)
            
        # print(value)
        for i in range(4):
            if self.ui.img_block[i].img is not None:
                ref_roi_img = get_roi_img(self.ui.img_block[0].img, self.ui.img_block[0].roi_coordinate)
                roi_img = get_roi_img(self.ui.img_block[i].img, self.ui.img_block[i].roi_coordinate)
                # 以左上角為起點裁剪成相同大小
                h = min(ref_roi_img.shape[0], roi_img.shape[0])
                w = min(ref_roi_img.shape[1], roi_img.shape[1])

                ref_roi_img = ref_roi_img[:h, :w]
                roi_img = roi_img[:h, :w]
                # cv2.imshow("roi_img"+str(i), roi_img)
                # cv2.waitKey(0)
                self.ui.score[i][0].setText(str(get_perceptual_distance(ref_roi_img,  roi_img)))
                self.ui.img_block[0].setPhoto(ref_roi_img, self.ui.img_block[0].filename)
                self.ui.img_block[i].setPhoto(roi_img, self.ui.img_block[i].filename)
                self.ui.score_region[i].show()
        
