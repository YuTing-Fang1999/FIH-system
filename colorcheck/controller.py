from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import numpy as np

import sys
sys.path.append("..")
from .UI import Ui_MainWindow
from .SNR_window import SNR_window
from myPackage.selectROI_window import SelectROI_window

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.filefolder = './'
        self.default_ROI = None

        self.selectROI_window = []
        self.SNR_window = []
        for i in range(4): self.SNR_window.append(SNR_window(tab_idx = i))
        for i in range(4): self.selectROI_window.append(SelectROI_window(tab_idx = i))

        self.setup_control()

    def setup_event(self, i):
        self.ui.open_img_btn[i].clicked.connect(lambda : self.open_img(self.ui.img_block[i], i))
        
    def setup_control(self):
        self.setup_event(0) # 須個別賦值(不能用for迴圈)，否則都會用到同一個數值
        self.setup_event(1)
        self.setup_event(2)
        self.setup_event(3)
        # self.ui.btn_same_ROI.clicked.connect(lambda : self.compute(same_ROI = True)) 
        self.ui.btn_compute.clicked.connect(lambda : self.compute()) 

        # self.ui.rubberBand[0].setGeometry(QtCore.QRect(QtCore.QPoint(0,0), QtCore.QPoint(100,100)))  # QSize() 此時爲-1 -1
        # self.ui.rubberBand[0].show()
        
    def open_img(self, img_block, tab_idx):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  self.filefolder, # start path
                  'Image Files(*.png *.jpg *.jpeg *.bmp)')    
        
        if filename == '': return
        self.filefolder = '/'.join(filename.split('/')[:-1])
        
        # load img
        img = cv2.imdecode( np.fromfile( file = filename, dtype = np.uint8 ), cv2.IMREAD_COLOR )
        img_block.ROI.set_img(img, img_block)
        img_block.setFocus()
        
        self.ui.tabWidget.setCurrentIndex(tab_idx)

    def compute(self):
        cv2.destroyAllWindows()
        for w in self.SNR_window: w.close()

        img_idx = []
        for i in range(4):
            if self.ui.img_block[i].ROI.img is not None: img_idx.append(i)
        if(len(img_idx) < 1):
            QMessageBox.about(self, "info", "至少要load一張圖片")
            return False

        roi_idx = []
        for i in img_idx:
            if self.ui.img_block[i].ROI.roi_img is not None: roi_idx.append(i)

        if(len(roi_idx) < 1):
            QMessageBox.about(self, "info", "未選擇區域")
            return False

        if roi_idx != img_idx:
            roi_idx = roi_idx[0]
            for i in img_idx:
                if i != roi_idx:

                    self.ui.img_block[i].ROI.set_x1_y1_x2_y2(self.ui.img_block[roi_idx].ROI.get_x1_y1_x2_y2())
                    self.ui.img_block[i].ROI.setRubberBandGeometry()

                    img_roi = self.ui.img_block[i].ROI.get_ROI()
                    if img_roi is None: return
                    self.ui.img_block[i].ROI.roi = img_roi
                    self.ui.img_block[i].ROI.roi_coordinate = self.ui.img_block[roi_idx].ROI.roi_coordinate
        
        # 顯示圖片
        all_SNR = []
        for i in img_idx:
            cv2.imshow('PIC'+str(i+1), self.ui.img_block[i].ROI.get_rectangle_img_by_roi_coordinate())
            # cv2.resizeWindow('PIC'+str(i+1), 200, 200)
            cv2.moveWindow('PIC'+str(i+1), 0, 200*i)
            cv2.waitKey(100)

            all_SNR.append(self.get_SNR(self.ui.img_block[i]))

        max_val = np.max(all_SNR, axis=0)
        min_val = np.min(all_SNR, axis=0)
        idx = 0
        for i in img_idx:
            self.SNR_window[i].set_SNR(all_SNR[idx], max_val, min_val)
            self.SNR_window[i].show()
            idx+=1

    def get_SNR(self, img_block):
        rois = img_block.ROI.get_roi_img_by_roi_coordinate()
        SNR = [self.compute_SNR(patch) for patch in rois]
        return SNR

    def compute_SNR(self, patch):
        Y = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        R = patch[:,:,2]
        G = patch[:,:,1]
        B = patch[:,:,0]
        YSNR = 20*np.log10(self.signal_to_noise(Y))
        RSNR = 20*np.log10(self.signal_to_noise(R))
        GSNR = 20*np.log10(self.signal_to_noise(G))
        BSNR = 20*np.log10(self.signal_to_noise(B))

        return [np.around(YSNR, 3), np.around(RSNR, 3), np.around(GSNR, 3), np.around(BSNR, 3), np.around(np.mean([YSNR, RSNR, GSNR, BSNR]), 3)]

    def signal_to_noise(self, a):
        a = np.asanyarray(a)
        m = a.mean()
        sd = a.std()
        # print(m)
        # print(sd)
        if sd < 1e-9: sd = 1e-9
        # print(m/sd)
        # print()
        return m/sd



