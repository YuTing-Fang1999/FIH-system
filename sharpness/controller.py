from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from scipy.signal import convolve2d
import math

from .UI import Ui_MainWindow

class ROI:
    # 建構式
    def __init__(self):
        self.y_start = 0
        self.x_start = 0
        self.y_end = 0
        self.x_end = 0
        self.img = None
        self.img_roi = None
        self.label = None
        
    # 方法(Method)
    def set_x1_y1(self, x, y):
        self.x1 = x
        self.y1 = y

    def set_x2_y2(self, x, y):
        self.x2 = x
        self.y2 = y
    
    def set_img(self, img = None, label = None):
        if isinstance(img, np.ndarray):
            self.img_roi = None
            self.img = img
            self.label = label
            

        if isinstance(self.img, np.ndarray):
            qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*self.img.shape[2], QImage.Format_RGB888).rgbSwapped()
            self.label.setPixmap(QPixmap(qimg))
    
    def get_Imatest_anysharp(self):
        P = self.img_roi.copy()
        P = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY).astype('float64')
        P /= 255 # linearized

        sobelx = cv2.Sobel(P, cv2.CV_16S, 1, 0)  # x方向梯度 ksize默認為3x3
        sobely = cv2.Sobel(P, cv2.CV_16S, 0, 1)  # y方向梯度

        abx = np.abs(sobelx)  # 取sobelx絕對值
        aby = np.abs(sobely)  # 取sobely絕對值

        sx = np.mean(abx)/np.mean(P)
        sy = np.mean(aby)/np.mean(P)
        
        sTotal = np.sqrt(sx*sx + sy*sy)

        return np.round(sTotal, 4)
        
        # # https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
        # im = cv2.cvtColor(self.img_roi.copy(), cv2.COLOR_BGR2GRAY) # to grayscale
        # array = np.asarray(im, dtype=np.int32)

        # gy, gx = np.gradient(array)
        # gnorm = np.sqrt(gx**2 + gy**2)
        # sharpness = np.average(gnorm)

        # return sharpness


    def get_sharpness(self):
        I = self.img_roi
        var = cv2.Laplacian(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        return np.round(math.sqrt(var), 4)
    
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
        self.img_roi = None
        if self.img is None: return None

        label_w = self.label.width()
        label_h = self.label.height()

        self.w = abs(self.x1-self.x2)
        self.h = abs(self.y1-self.y2)
#       print(self.w, self.h)
        if self.w==0 or self.h==0 : return None
        
        if(self.x1 > self.x2):
            self.x1 = self.x2
            self.y1 = self.y2
            
        if(self.img.shape[0]/self.img.shape[1] < label_h/label_w):
            y_start = int(self.y1 - label_h/2)
            x_start = self.x1
            x_end = x_start + self.w
            y_end = y_start + self.h
            
            factor = self.img.shape[1]/label_w
            x_start = int(x_start*factor)
            y_start = int(y_start*factor)
            x_end = int(x_end*factor)
            y_end = int(y_end*factor)
            
            y_start = int(y_start + self.img.shape[0]/2)
            y_end = int(y_end + self.img.shape[0]/2)
        else:
            x_start = int(self.x1 - label_w/2)
            y_start = self.y1
            x_end = x_start + self.w
            y_end = y_start + self.h
            
            factor = self.img.shape[0]/label_h
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

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.filefolder = './'

        self.myROI = []
        for i in range(4):
            self.myROI.append(ROI())

        self.setup_control()

    def setup_event(self, i):
        self.ui.open_img_btn[i].clicked.connect(lambda : self.open_img(self.ui.img_block[i], i))
        # # set_clicked_position
        self.ui.img_block[i].mousePressEvent = lambda event : self.show_mouse_press(event, self.ui.rubberBand[i], self.myROI[i])
        self.ui.img_block[i].mouseMoveEvent = lambda event : self.show_mouse_move(event, self.ui.rubberBand[i])
        self.ui.img_block[i].mouseReleaseEvent = lambda event : self.show_mouse_release(event, self.ui.rubberBand[i], self.myROI[i], self.ui.img_block[i])

    def setup_control(self):
        self.setup_event(0) # 須個別賦值(不能用for迴圈)，否則都會用到同一個數值
        self.setup_event(1)
        self.setup_event(2)
        self.setup_event(3)
        self.ui.btn_same_ROI.clicked.connect(lambda : self.compute(same_ROI = True)) 
        self.ui.btn_dif_ROI.clicked.connect(lambda : self.compute(same_ROI = False)) 
        
    def open_img(self, img_block, tab_idx):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  self.filefolder, # start path
                  'Image Files(*.png *.jpg *.jpeg *.bmp)')    
        
        if filename == '': return
        self.filefolder = '/'.join(filename.split('/')[:-1])
        
        # load img
        img = cv2.imdecode( np.fromfile( file = filename, dtype = np.uint8 ), cv2.IMREAD_COLOR )
        self.myROI[tab_idx].set_img(img, img_block)
        self.ui.rubberBand[tab_idx].hide()
        
        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*img.shape[2], QImage.Format_RGB888).rgbSwapped()
        img_block.setPixmap(QPixmap(qimg))
        self.ui.tabWidget.setCurrentIndex(tab_idx)

    def compute(self, same_ROI):
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
        if(len(roi_idx) == 0):
            QMessageBox.about(self, "info", "還未選擇區域")
            return False
        elif (len(roi_idx) > 1 and same_ROI):
            QMessageBox.about(self, "info", "選了不只一個區域")
            return False

        if not same_ROI and len(img_idx) != len(roi_idx):
            dif = list(set(img_idx) - set(roi_idx))
            dif = ["PIC"+str(idx+1) for idx in dif]
            QMessageBox.about(self, "info", " ".join(dif)+"還未選擇區域")
            return False

        roi_idx = roi_idx[0]

        if same_ROI:
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
        for i in img_idx:
            cv2.imshow('PIC'+str(i+1), self.myROI[i].img_roi)
            # cv2.resizeWindow('PIC'+str(i+1), 200, 200)
            cv2.moveWindow('PIC'+str(i+1), 0, 200*i)
            cv2.waitKey(100)

            self.ui.score[i][0].setText(str(self.myROI[i].get_sharpness()))
            self.ui.score[i][1].setText(str(self.myROI[i].get_noise()))
            self.ui.score[i][2].setText(str(self.myROI[i].get_Imatest_anysharp()))
        
    def show_mouse_press(self, event, rubberBand, ROI):
        # print(f"[show_mouse_press] {event.x()=}, {event.y()=}, {event.button()=}")
        self.origin_pos = event.pos()
        ROI.set_x1_y1(event.x(), event.y())
    
        rubberBand.setGeometry(QtCore.QRect(self.origin_pos, QtCore.QSize()))  # QSize() 此時爲-1 -1
        rubberBand.show()

    def show_mouse_move(self, event, rubberBand):
#         print(f"[show_mouse_move] {event.x()=}, {event.y()=}, {event.button()=}")
        rubberBand.setGeometry(QtCore.QRect(self.origin_pos, event.pos()).normalized())  # 這裏可以

    def show_mouse_release(self, event, rubberBand, ROI, label):
#         print(f"[show_mouse_release] {event.x()=}, {event.y()=}, {event.button()=}")

        ROI.set_x2_y2(event.x(), event.y())
        img_roi = ROI.get_ROI()
        if img_roi is None: 
            rubberBand.hide()
            ROI.img_roi = None
        else: 
            ROI.img_roi = img_roi
        rect = rubberBand.geometry()


