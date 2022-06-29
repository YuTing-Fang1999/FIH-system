# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5.Qt import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QSize
import numpy as np

import sys
sys.path.append("..")
from myPackage.ImageViewer import ImageViewer

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.rubberBand = []
        self.open_img_btn = []
        self.img_block = []
        self.score = []

        spacerItem = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(992, 762)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayout_parent = QtWidgets.QVBoxLayout(self.centralwidget)

        self.horizontalLayout_btn = QtWidgets.QHBoxLayout()
        self.horizontalLayout_btn.addItem(spacerItem)
        for i in range(4):
            open_img_btn = QtWidgets.QPushButton(self.centralwidget)
            open_img_btn.setText("Load Pic"+str(i+1))
            self.open_img_btn.append(open_img_btn)
            self.horizontalLayout_btn.addWidget(open_img_btn)
        self.btn_compute = QtWidgets.QPushButton(self.centralwidget)
        self.btn_compute.setText("Compute")
        self.horizontalLayout_btn.addWidget(self.btn_compute)
        self.horizontalLayout_btn.addItem(spacerItem)

        self.verticalLayout_parent.addLayout(self.horizontalLayout_btn)

        self.horizontalLayout_imgTab = QtWidgets.QHBoxLayout()
        # self.horizontalLayout_imgTab.addItem(spacerItem)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        for i in range(4):
            tab = QtWidgets.QWidget()
            img_block = ImageViewer(i, function="colorcheck")
            img_block.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                            "border-color: rgb(0, 0, 0);\n"
                                            "border: 2px solid;")
            img_block.setAlignment(QtCore.Qt.AlignCenter)
            # img_block.setText("PIC"+str(i+1))

            tab_wraper = QtWidgets.QVBoxLayout(tab)
            tab_wraper.addWidget(img_block)
            self.img_block.append(img_block)
            self.tabWidget.addTab(tab, "PIC"+str(i+1))
            self.rubberBand.append(QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, img_block))
        self.horizontalLayout_imgTab.addWidget(self.tabWidget)
        # self.horizontalLayout_imgTab.addItem(spacerItem)
        self.horizontalLayout_imgTab.setStretch(0, 1)
        self.horizontalLayout_imgTab.setStretch(1, 5)
        self.horizontalLayout_imgTab.setStretch(2, 1)

        self.verticalLayout_parent.addLayout(self.horizontalLayout_imgTab)
        
        

        self.verticalLayout_parent.setStretch(1, 2)
        self.verticalLayout_parent.setStretch(2, 1)

        MainWindow.setStyleSheet(
                        # "background-color: rgb(66, 66, 66);\n"
                        "QLabel{font-size:12pt; font-family:微軟正黑體;}"
                        "QPushButton{font-size:12pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 992, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
