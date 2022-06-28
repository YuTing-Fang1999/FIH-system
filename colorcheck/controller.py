from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

from .UI import Ui_MainWindow
from .SNR_window import SNR_window

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.origin_pos = None
        self.filefolder = './'
        self.myROI = []
        self.default_ROI = None
        self.SNR_window = []
        for i in range(4):
            self.SNR_window.append(SNR_window(tab_idx = i))

        self.setup_control()

    def setup_event(self, i):
        # self.myROI.append(ROI(self.ui.img_block[i], self.ui.rubberBand[i]))

        self.ui.open_img_btn[i].clicked.connect(lambda : self.open_img(self.ui.img_block[i], i))
        # # set_clicked_position
        # self.ui.img_block[i].mousePressEvent = lambda event : self.show_mouse_press(event, self.ui.rubberBand[i], self.myROI[i], self.ui.img_block[i])
        # self.ui.img_block[i].mouseMoveEvent = lambda event : self.show_mouse_move(event, self.ui.rubberBand[i], self.ui.img_block[i])
        # self.ui.img_block[i].mouseReleaseEvent = lambda event : self.show_mouse_release(event, self.ui.rubberBand[i], self.myROI[i], self.ui.img_block[i], i+1)

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
        img_block.ROI.set_img(img, img_block, self.default_ROI)
        
        self.ui.tabWidget.setCurrentIndex(tab_idx)

    def compute(self):
        cv2.destroyAllWindows()

        img_idx = []
        for i in range(4):
            if self.myROI[i].img is not None: img_idx.append(i)
        if(len(img_idx) < 1):
            QMessageBox.about(self, "info", "至少要load一張圖片")
            return False

        roi_idx = []
        for i in img_idx:
            if self.myROI[i].img_roi is not None: roi_idx.append(i)

        if(len(roi_idx) < 1):
            QMessageBox.about(self, "info", "未選擇區域")
            return False

        if roi_idx != img_idx:
            roi_idx = roi_idx[0]
            for i in img_idx:
                if i != roi_idx:
                    self.ui.rubberBand[i].setGeometry(QtCore.QRect(
                        self.myROI[roi_idx].x1, self.myROI[roi_idx].y1, self.myROI[roi_idx].x2-self.myROI[roi_idx].x1, self.myROI[roi_idx].y2-self.myROI[roi_idx].y1
                    ).normalized())
                    self.ui.rubberBand[i].show()

                    self.myROI[i].x1 = self.myROI[roi_idx].x1
                    self.myROI[i].y1 = self.myROI[roi_idx].y1
                    self.myROI[i].x2 = self.myROI[roi_idx].x2 
                    self.myROI[i].y2 = self.myROI[roi_idx].y2

                    roi = self.myROI[i].get_ROI()
                    if roi is None: return
                    self.myROI[i].roi = roi
        
        # 顯示圖片
        all_SNR = []
        for i in img_idx:
            cv2.imshow('PIC'+str(i+1), self.draw_24_block(self.myROI[i].img_roi))
            # cv2.resizeWindow('PIC'+str(i+1), 200, 200)
            cv2.moveWindow('PIC'+str(i+1), 0, 200*i)
            cv2.waitKey(100)

            all_SNR.append(self.get_SNR(self.myROI[i].img_roi))

        max_val = np.max(all_SNR, axis=0)
        min_val = np.min(all_SNR, axis=0)
        idx = 0
        for i in img_idx:
            self.SNR_window[i].set_SNR(all_SNR[idx], max_val, min_val)
            self.SNR_window[i].show()
            idx+=1
        
#     def show_mouse_press(self, event, rubberBand, ROI):
#         # print(f"[show_mouse_press] {event.x()=}, {event.y()=}, {event.button()=}")
#         self.origin_pos = event.pos()
#         # print(event.pos())
#         ROI.set_x1_y1(event.x(), event.y())
#         ROI.img_roi = None
    
#         rubberBand.setGeometry(QtCore.QRect(self.origin_pos, QtCore.QSize()))  # QSize() 此時爲-1 -1
#         rubberBand.show()
#         cv2.destroyAllWindows()

#     def show_mouse_move(self, event, rubberBand):
#         # print(f"[show_mouse_move] {event.x()=}, {event.y()=}, {event.button()=}")
#         # print(event.pos())
#         if self.origin_pos:
#             rubberBand.setGeometry(QtCore.QRect(self.origin_pos, event.pos()).normalized())  # 這裏可以

#     def show_mouse_release(self, event, rubberBand, ROI, tab_idx):
# #         print(f"[show_mouse_release] {event.x()=}, {event.y()=}, {event.button()=}")

#         ROI.set_x2_y2(event.x(), event.y())
#         img_roi = ROI.get_ROI()
#         if img_roi is None: 
#             rubberBand.hide()
#             ROI.img_roi = None
#         else: 
#             cv2.destroyAllWindows()
#             img = img_roi.copy()
#             cv2.imshow('roi '+str(tab_idx), self.draw_24_block(img))
#             cv2.waitKey(100)

#             self.default_ROI = [ROI.x1, ROI.y1, ROI.x2, ROI.y2]

    
    
    
    

    def get_SNR(self, img):
        SNR = []

        h, w, c = img.shape
        square = 0.08*w
        padding = 0.087*w

        start_h = 0.044*w
        for i in range(4):
            start_w = 0.044*w
            for j in range(6):
                patch = img[int(start_h):int(start_h+square), int(start_w):int(start_w+square)]
                SNR.append(self.compute_SNR(patch))
                # cv2.rectangle(img, (int(start_w), int(start_h)), (int(start_w+square), int(start_h+square)), color, thickness)
                start_w+=(square+padding)
            start_h+=(square+padding)
        # print(SNR)
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



