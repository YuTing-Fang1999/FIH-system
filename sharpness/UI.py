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

import sys
sys.path.append("..")
from myPackage.ImageViewer import ImageViewer

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.open_img_btn = []
        self.img_block = []
        self.score = []

        spacerItem = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(992, 762)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayout_parent = QtWidgets.QVBoxLayout(self.centralwidget)

        self.horizontalLayout_upper = QtWidgets.QHBoxLayout()
        self.horizontalLayout_upper.addItem(spacerItem)
        for i in range(4):
            open_img_btn = QtWidgets.QPushButton(self.centralwidget)
            open_img_btn.setText("Load Pic"+str(i+1))
            self.open_img_btn.append(open_img_btn)
            self.horizontalLayout_upper.addWidget(open_img_btn)
        self.horizontalLayout_upper.addItem(spacerItem)
        self.verticalLayout_parent.addLayout(self.horizontalLayout_upper)

        self.horizontalLayout_medium = QtWidgets.QHBoxLayout()
        # self.horizontalLayout_medium.addItem(spacerItem)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        for i in range(4):
            tab = QtWidgets.QWidget()
            img_block = ImageViewer()
            img_block.setAlignment(QtCore.Qt.AlignCenter)

            tab_wraper = QtWidgets.QVBoxLayout(tab)
            tab_wraper.addWidget(img_block)
            self.img_block.append(img_block)
            self.tabWidget.addTab(tab, "PIC"+str(i+1))
        self.horizontalLayout_medium.addWidget(self.tabWidget)
        # self.horizontalLayout_medium.addItem(spacerItem)
        # self.horizontalLayout_medium.setStretch(0, 1)
        # self.horizontalLayout_medium.setStretch(1, 5)
        # self.horizontalLayout_medium.setStretch(2, 1)

        self.verticalLayout_parent.addLayout(self.horizontalLayout_medium)

        # self.horizontalLayout_lower = QtWidgets.QHBoxLayout()
        # "gnorm", "Imatest",
        # name = ["sharpness", "noise", "Imatest\nSobel", "Imatest\nLaplacian", "H", "V"]
        # for i in range(4):
        #     gridLayout = QtWidgets.QGridLayout()
        #     label = QtWidgets.QLabel(self.centralwidget)
        #     label.setText("PIC"+str(i+1))
        #     gridLayout.addWidget(label, 0, 0)
        #     score = []
        #     for j in range(len(name)):
        #         label = QtWidgets.QLabel(self.centralwidget)
        #         label.setText(name[j])
        #         gridLayout.addWidget(label, j+1, 0)
        #         label = QtWidgets.QLabel(self.centralwidget)
        #         score.append(label)
        #         gridLayout.addWidget(label, j+1, 1)
        #     self.score.append(score)
        #     self.horizontalLayout_lower.addLayout(gridLayout)
        # self.horizontalLayout_lower.setSpacing(50)
        # self.verticalLayout_parent.addLayout(self.horizontalLayout_lower)

        # self.verticalLayout_parent.setStretch(1, 2)
        # self.verticalLayout_parent.setStretch(2, 1)

        MainWindow.setStyleSheet(
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
