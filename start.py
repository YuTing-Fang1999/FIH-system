# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
# pyuic5 -x UI.ui -o UI.py

from PyQt5 import QtCore, QtWidgets
from fft.controller import MainWindow_controller as fft_window
from colorcheck.controller import MainWindow_controller as colorcheck_window
from sharpness.controller import MainWindow_controller as sharpness_window
from dxo_dead_leaves.controller import MainWindow_controller as dxo_dead_leaves_window
from perceptual_distance.controller import MainWindow_controller as perceptual_distance_window
from myPackage.selectROI_window import SelectROI_window
import json
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.setting = self.read_setting()
        self.selectROI_window = SelectROI_window(self.setting["filefolder"])
        self.selectROI_window.update_filefolder_signal.connect(self.update_filefolder)
        MainWindow.resize(400, 0)
        centralwidget = QtWidgets.QWidget(MainWindow)
        verticalLayout_parent = QtWidgets.QVBoxLayout(centralwidget)

        self.windows = []

        name = ["頻譜分析", "colorcheck", "sharpness/noise", "dxo_dead_leaves", "perceptual_distance"]
        self.pushButton = []
        for i in range(len(name)):
            pushButton = QtWidgets.QPushButton(centralwidget)
            pushButton.setText(name[i])
            verticalLayout_parent.addWidget(pushButton)
            self.pushButton.append(pushButton)

        MainWindow.setCentralWidget(centralwidget)
        MainWindow.setStyleSheet("QMainWindow {background-color: rgb(66, 66, 66);}"
                                 """
                                QPushButton {
                                    font-size:14pt; 
                                    font-family:微軟正黑體; 
                                    font-weight: bold; 
                                    letter-spacing: 4pt;
                                    background-color:rgb(255, 170, 0);
                                }
                                """
                                 )

        # Sub Window
        # self.fft_window = fft_window()
        # self.colorcheck_window = colorcheck_window()
        # self.sharpness_window = sharpness_window()
        # self.dxo_dead_leaves = dxo_dead_leaves_window()

        # Button Event
        # self.pushButton[0].clicked.connect(self.fft_window.showMaximized)
        # self.pushButton[1].clicked.connect(self.colorcheck_window.showMaximized)
        # self.pushButton[2].clicked.connect(self.sharpness_window.showMaximized)
        # self.pushButton[3].clicked.connect(self.dxo_dead_leaves.showMaximized)

        self.pushButton[0].clicked.connect(self.show_fft_window)
        self.pushButton[1].clicked.connect(self.show_colorcheck_window)
        self.pushButton[2].clicked.connect(self.show_sharpness_window)
        self.pushButton[3].clicked.connect(self.show_dxo_dead_leaves_window)
        self.pushButton[4].clicked.connect(self.show_perceptual_distance_window)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        MainWindow.closeEvent = self.closeEvent
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    def show_fft_window(self):
        w = fft_window(self.selectROI_window)
        self.windows.append(w)
        w.showMaximized()

    def show_colorcheck_window(self):
        w = colorcheck_window(self.selectROI_window)
        self.windows.append(w)
        w.showMaximized()

    def show_sharpness_window(self):
        w = sharpness_window(self.selectROI_window)
        self.windows.append(w)
        w.showMaximized()

    def show_dxo_dead_leaves_window(self):
        w = dxo_dead_leaves_window(self.selectROI_window)
        self.windows.append(w)
        w.showMaximized()

    def show_perceptual_distance_window(self):
        w = perceptual_distance_window(self.selectROI_window)
        self.windows.append(w)
        w.showMaximized()

    def update_filefolder(self, filefolder):
        if filefolder != "./":
            self.setting["filefolder"] = filefolder

    def read_setting(self):
        if os.path.exists('setting.json'):
            with open('setting.json', 'r') as f:
                setting = json.load(f)
                if not os.path.exists(setting["filefolder"]):
                    setting["filefolder"] = "./"
                return setting
            
        else:
            print("找不到設定檔，重新生成一個新的設定檔")
            return {
                "filefolder": "./"
            }

    def write_setting(self):
        print('write_setting')
        with open("setting.json", "w") as outfile:
            outfile.write(json.dumps(self.setting, indent=4))

    def closeEvent(self, e):
        self.write_setting()


if __name__ == "__main__":
    import sys
    # 高分辨率屏幕自適應
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
