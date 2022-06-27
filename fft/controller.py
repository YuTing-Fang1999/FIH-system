from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .UI import Ui_MainWindow

import time
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
            self.img = img
            self.label = label

        if isinstance(self.img, np.ndarray):
            qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*self.img.shape[2], QImage.Format_RGB888).rgbSwapped()
            qimg = QPixmap(qimg)
            if self.img.shape[1]/self.img.shape[0] > self.label.width()/self.label.height():
                qimg = qimg.scaledToWidth(self.label.width())
            else:
                qimg = qimg.scaledToHeight(self.label.height())
            self.label.setPixmap(qimg)

    def get_sharpness(self, I):
        return np.round(cv2.Laplacian(cv2.cvtColor(I, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()/100, 4)
    
    def get_noise(self, I):  
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        H, W = I.shape

        M = [[1, -2, 1],
           [-2, 4, -2],
           [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

        return np.round(sigma, 4)
    
    def get_ROI(self, label_w, label_h):
        self.label_w = label_w
        self.label_h = label_h

        self.img_roi = None
        if self.img is None: return None

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

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=20, height=4, dpi=80):
        self.fig = Figure(figsize=(32, 8), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.filefolder = './'

        # 註冊 rubberBand
        self.ui.rubberBand1 = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.ui.img_block)
        self.ui.rubberBand2 = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.ui.img_block_2)

        self.myROI1 = ROI()
        self.myROI2 = ROI()

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.

        self.his_canvas = MplCanvas()
        self.his_layout = QtWidgets.QHBoxLayout(self.ui.his)
        self.his_canvas_2 = MplCanvas()
        self.his_layout_2 = QtWidgets.QHBoxLayout(self.ui.his_2)

        self.fft_his_canvas = MplCanvas()
        self.fft_his_layout = QtWidgets.QHBoxLayout(self.ui.fft_his)
        self.fft_his_canvas_2 = MplCanvas()
        self.fft_his_layout_2 = QtWidgets.QHBoxLayout(self.ui.fft_his_2)

        # self.psd_his_canvas = MplCanvas()
        # self.psd_his_layout = QtWidgets.QHBoxLayout(self.ui.psd_his)
        # self.psd_his_canvas_2 = MplCanvas()
        # self.psd_his_layout_2 = QtWidgets.QHBoxLayout(self.ui.psd_his_2)

    def setup_control(self):
        self.ui.open_img_btn.clicked.connect(
            lambda: self.open_img(
                self.ui.img_block, 0
            )
        )
        
        self.ui.open_img_btn_2.clicked.connect(
            lambda: self.open_img(
                self.ui.img_block_2, 1
            )
        ) 

        # self.ui.btn_all.clicked.connect(lambda: self.put_to_chart(isROI=False))
        self.ui.btn_roi.clicked.connect(lambda: self.put_to_chart(isROI=True))

        # set_clicked_position
        self.ui.img_block.mousePressEvent = lambda event : self.show_mouse_press(event, self.ui.rubberBand1, self.myROI1)
        self.ui.img_block.mouseMoveEvent = lambda event : self.show_mouse_move(event, self.ui.rubberBand1)
        self.ui.img_block.mouseReleaseEvent = lambda event : self.show_mouse_release(event, self.ui.rubberBand1, self.myROI1, self.ui.img_block)

        self.ui.img_block_2.mousePressEvent = lambda event : self.show_mouse_press(event, self.ui.rubberBand2, self.myROI2)
        self.ui.img_block_2.mouseMoveEvent = lambda event : self.show_mouse_move(event, self.ui.rubberBand2)
        self.ui.img_block_2.mouseReleaseEvent = lambda event : self.show_mouse_release(event, self.ui.rubberBand2, self.myROI2, self.ui.img_block_2)
    
        self.ui.resizeEvent = lambda event : self.resizeEvent(event)

    def resizeEvent(self, event):
        print('resizeEvent')
        self.myROI1.img_roi = None
        self.ui.rubberBand1.hide()
        self.myROI2.img_roi = None
        self.ui.rubberBand2.hide()


    def open_img(self, img_block, tab_idx):
        filename = self.get_file_path()
        if filename == '': return
        self.filefolder = '/'.join(filename.split('/')[:-1])
        
        img = cv2.imdecode( np.fromfile( file = filename, dtype = np.uint8 ), cv2.IMREAD_COLOR ) 

        if tab_idx == 0: 
            self.myROI1.img_roi = None
            self.myROI1.set_img(img, img_block)
            self.ui.rubberBand1.hide()
            
        elif tab_idx == 1: 
            self.myROI2.img_roi = None
            self.myROI2.set_img(img, img_block)
            self.ui.rubberBand2.hide()

    def get_file_path(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  self.filefolder, # start path
                  'Image Files(*.png *.jpg *.jpeg *.bmp)')                 
        return filename
    
    def get_fft(self, img):
        # to gray
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        # to fft
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        img = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

        return img
    
    def set_fft_img(self, img, label):
        # clip
        img = np.around(img)
        img = np.clip(img,0,255)
        img = np.array(img,np.uint8)
        # set img
        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Indexed8)
        label.setPixmap(QPixmap(qimg))

    def set_img_his(self, img, canvas, layout, maxX):
        hist,bins = np.histogram(img, bins=maxX, range=(0,maxX))
        bins = bins[:-1]
        self.set_his(hist, bins, canvas, layout)

    def azimuthalAverage(self, image, center=None):
        """
        Calculate the azimuthally averaged radial profile.

        image - The 2D image
        center - The [x,y] pixel coordinates used as the center. The default is 
                None, which then uses the center of the image (including 
                fracitonal pixels).
        
        """
        # Calculate the indices from the image
        y, x = np.indices(image.shape) #shape與圖片相同，y[0,0]代表image[0,0]的y座標
        if not center:
            center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])

        r = np.hypot(x - center[0], y - center[1]) #找出每個位置距離圓心的半徑
        # Get sorted radii
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]

        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)

        # Find all pixels that fall within each radial bin.
        deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
        rind = np.where(deltar)[0]       # location of changed radius
        nr = rind[1:] - rind[:-1]        # number of radius bin
        
        # Cumulative sum to figure out sums for each radius bin
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]

        radial_prof = tbin / nr

        return radial_prof

    def set_psd_his(self, psd2D, canvas, layout):
        # Calculate a 2D power spectrum
        # psd2D = np.abs( fft )**2

        # Calculate the azimuthally averaged 1D power spectrum
        print(psd2D.shape)
        psd1D = self.azimuthalAverage(psd2D)
        print(psd1D.shape)

        # psd1D = np.log10(psd1D)

        self.set_his(psd1D, range(len(psd1D)), canvas, layout)
        
    def set_his(self, hist, bins, canvas, layout): # histogram
        canvas.axes.cla() 
        canvas.axes.bar(bins,hist)
#         canvas.axes.axis(ymin=0,ymax=maxY)
        canvas.fig.canvas.draw() # 這裡注意是畫布重繪，self.figs.canvas
        canvas.fig.canvas.flush_events() # 畫布刷新self.figs.canvas
        layout.addWidget(canvas)

    
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
        img_roi = ROI.get_ROI(label.width(), label.height())
        if img_roi is None: 
            rubberBand.hide()
            ROI.img_roi = None
        else: 
            ROI.img_roi = img_roi
        rect = rubberBand.geometry()


    def put_to_chart(self, isROI=False):
        cv2.destroyAllWindows()

        img1 = self.myROI1.img
        img2 = self.myROI2.img 
        if img1 is None or img2 is None: return

        if isROI:
            img1 = self.myROI1.img_roi
            img2 = self.myROI2.img_roi

            if img1 is None and img2 is not None: 
                self.ui.rubberBand1.setGeometry(QtCore.QRect(
                    self.myROI2.x1, self.myROI2.y1, self.myROI2.x2-self.myROI2.x1, self.myROI2.y2-self.myROI2.y1
                ).normalized())
                self.ui.rubberBand1.show()
                
                self.myROI1.x1 = self.myROI2.x1
                self.myROI1.y1 = self.myROI2.y1
                self.myROI1.x2 = self.myROI2.x2 
                self.myROI1.y2 = self.myROI2.y2 
                img1 = self.myROI1.get_ROI(self.myROI2.label_w, self.myROI2.label_h)
                
            if img2 is None and img1 is not None: 
                self.ui.rubberBand2.setGeometry(QtCore.QRect(
                    self.myROI1.x1, self.myROI1.y1, self.myROI1.x2-self.myROI1.x1, self.myROI1.y2-self.myROI1.y1
                ).normalized())
                self.ui.rubberBand2.show()
                
                self.myROI2.x1 = self.myROI1.x1
                self.myROI2.y1 = self.myROI1.y1
                self.myROI2.x2 = self.myROI1.x2 
                self.myROI2.y2 = self.myROI1.y2 
                img2 = self.myROI2.get_ROI(self.myROI1.label_w, self.myROI1.label_h)
        
            if img1 is None or img2 is None: return

        fft1 = self.get_fft(img1)
        fft2 = self.get_fft(img2)

        self.set_img_his(img1, self.his_canvas, self.his_layout, 256)
        # self.set_psd_his(fft1, self.psd_his_canvas, self.psd_his_layout)
        self.set_img_his(fft1, self.fft_his_canvas, self.fft_his_layout, 350)
        self.set_fft_img(fft1, self.ui.fft_block)

        self.set_img_his(img2, self.his_canvas_2, self.his_layout_2, 256)
        # self.set_psd_his(fft2, self.psd_his_canvas_2, self.psd_his_layout_2)
        self.set_img_his(fft2, self.fft_his_canvas_2, self.fft_his_layout_2, 350)
        self.set_fft_img(fft2, self.ui.fft_block_2)
        
#         # 顯示圖片
        cv2.imshow('My Image', img1)
#         cv2.resizeWindow("My Image", 300, 300)
        cv2.moveWindow("My Image", 0, 100)
        cv2.waitKey(100)
        cv2.imshow('My Image2', img2)     
#         cv2.resizeWindow("My Image2", 300, 300)
        cv2.moveWindow("My Image2", 0, 400)
        cv2.waitKey(100)
        

        
        
    
