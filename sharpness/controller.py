
from PyQt5 import QtWidgets
from .UI import Ui_MainWindow
import sys
sys.path.append("..")
from myPackage.selectROI_window import SelectROI_window

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.selectROI_window = SelectROI_window()
        self.setup_control()

    def setup_control(self):
        # 須個別賦值(不能用for迴圈)，否則都會用到同一個數值
        self.ui.open_img_btn[0].clicked.connect(lambda : self.open_img(0))
        self.ui.open_img_btn[1].clicked.connect(lambda : self.open_img(1))
        self.ui.open_img_btn[2].clicked.connect(lambda : self.open_img(2))
        self.ui.open_img_btn[3].clicked.connect(lambda : self.open_img(3))
        self.selectROI_window.to_main_window_signal.connect(self.set_roi_coordinate)
        
    def open_img(self, tab_idx):
        self.selectROI_window.open_img(tab_idx)

    def set_roi_coordinate(self, img_idx, img, roi_coordinate, filename):
        # print(tab_idx, img, roi_coordinate)
        ROI = self.ui.img_block[img_idx].ROI
        ROI.set_roi_img(img, roi_coordinate)
        key = ["sharpness", "noise", "Imatest Sobel", "Imatest Laplacian", "H", "V"]
        value = [
            ROI.get_sharpness(),
            ROI.get_noise(),
            ROI.get_gamma_Sobel(),
            ROI.get_gamma_Laplacian(),
            ROI.get_H(),
            ROI.get_V()
        ]
        text = filename+"\n"
        for k,v in zip(key, value):
            text+=k
            text+=": "
            text+=str(v)
            text+="\n"
        self.ui.img_block[img_idx].setPhoto(ROI.roi_img, text)
        # self.compute(img_idx)

    def compute(self, img_idx):
        # 顯示圖片
        self.ui.score[img_idx][0].setText(str(self.ui.img_block[img_idx].ROI.get_sharpness()))
        self.ui.score[img_idx][1].setText(str(self.ui.img_block[img_idx].ROI.get_noise()))
        # self.ui.score[img_idx][2].setText(str(self.ui.img_block[img_idx].ROI.get_average_gnorm()))
        # self.ui.score[img_idx][3].setText(str(self.ui.img_block[img_idx].ROI.get_Imatest()))
        self.ui.score[img_idx][2].setText(str(self.ui.img_block[img_idx].ROI.get_gamma_Sobel()))
        self.ui.score[img_idx][3].setText(str(self.ui.img_block[img_idx].ROI.get_gamma_Laplacian()))
        self.ui.score[img_idx][4].setText(str(self.ui.img_block[img_idx].ROI.get_H()))
        self.ui.score[img_idx][5].setText(str(self.ui.img_block[img_idx].ROI.get_V()))


