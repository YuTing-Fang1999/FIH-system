from fft.controller import MainWindow_controller
# from colorcheck.controller import MainWindow_controller
# from sharpness.controller import MainWindow_controller

from PyQt5 import QtWidgets, QtCore
import sys
# 高分辨率屏幕自適應
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
app = QtWidgets.QApplication(sys.argv)
window = MainWindow_controller()
window.show()
sys.exit(app.exec_())

