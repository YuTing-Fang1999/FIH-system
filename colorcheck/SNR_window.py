# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class SNR_window(QtWidgets.QMainWindow):
    def __init__(self, tab_idx = 1):
        super().__init__()
        self.move(600+200*tab_idx, 100)

        self.SNR = None
        self.label_SNR = [None]*24

        self.setWindowTitle("PIC "+str(tab_idx+1))
        self.resize(500, 0)
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.gridLayout = QtWidgets.QGridLayout()

        name = ["Y-SNR(dB)", "R-SNR(dB)", "G-SNR(dB)", "B-SNR(dB)", "A-SNR(dB)"]
        for i in range(5):
            label = QtWidgets.QLabel(self.centralwidget)
            label.setText(name[i])
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.gridLayout.addWidget(label, 0, i+1, 1, 1)

        for i in range(24):
            label = QtWidgets.QLabel(self.centralwidget)
            label.setText(str(i+1))
            self.gridLayout.addWidget(label, i+1, 0, 1, 1)
            self.label_SNR[i] = []
            for j in range(5):
                label = QtWidgets.QLabel(self.centralwidget)
                label.setAlignment(QtCore.Qt.AlignCenter)
                # label.setText(str(self.SNR[i][j]))
                label.setToolTip("{}, {}".format(i+1, name[j]))
                self.label_SNR[i].append(label)
                self.gridLayout.addWidget(label, i+1, j+1, 1, 1)

        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout.setColumnStretch(3, 1)
        self.gridLayout.setColumnStretch(4, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.setCentralWidget(self.centralwidget)

        self.setStyleSheet("QMainWindow {background-color: rgb(66, 66, 66);}"
                                """
                                QLabel {
                                    font-size:10pt; font-family:微軟正黑體; font-weight: bold;
                                    color: white;
                                    border: 1px solid black;
                                }
                                """
                                """
                                QToolTip { 
                                    background-color: black; 
                                    border: black solid 1px
                                }
                                """
        )

    def set_SNR(self, SNR, max_val = None, min_val = None):
        for i in range(24):
            for j in range(5):
                self.label_SNR[i][j].setText(str(SNR[i][j]))
                if SNR[i][j] == max_val[i][j]:
                    self.label_SNR[i][j].setStyleSheet("color: lightgreen;")
                elif SNR[i][j] == min_val[i][j]:
                    self.label_SNR[i][j].setStyleSheet("color: red;")
                else: 
                    self.label_SNR[i][j].setStyleSheet("color: white;")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = SNR_window()
    w.set_SNR([range(5)]*24)
    w.show()
    sys.exit(app.exec_())
