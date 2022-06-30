from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem
import cv2
import numpy as np
from requests import delete

class GraphicItem(QGraphicsRectItem):
    def __init__(self, x1, x2, w, parent=None):
        super().__init__(parent)
        self.setPen(Qt.red)
        r = QRectF(x1, x2, w, w) #起始座標,長,寬
        self.setRect(r)
        self.setFlag(QGraphicsItem.ItemIsSelectable)  # ***设置图元是可以被选择的
        self.setFlag(QGraphicsItem.ItemIsMovable)     # ***设置图元是可以被移动的

class ImageViewer(QtWidgets.QGraphicsView):
    # mouse_release_signal = pyqtSignal(ROI_coordinate)
    
    def __init__(self, parent = None):
        super(ImageViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        # 设置view可以进行鼠标的拖拽选择
        self.setDragMode(self.RubberBandDrag)

        self.square_rate = 0.08
        self.padding_rate = 0.088
        self.start_h_rate = 0.038
        self.start_w_rate = 0.038
        self.roi_coordinate = None
        self.gen_roi_coordinate_rate()

    def hasPhoto(self):
        return not self._empty

    def resizeEvent(self, event):
        self.fitInView()

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, img):
        
        self.img = img

        qimg = QImage(np.array(img), img.shape[1], img.shape[0], img.shape[1]*img.shape[2], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qimg)

        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()
    
    def delete_all_item(self):
        for item in self._scene.items()[:-1]:
            self._scene.removeItem(item)

    def gen_roi_coordinate_rate(self):
        self.roi_coordinate_rate = []
        square = self.square_rate
        padding = self.padding_rate

        start_h = self.start_h_rate
        for i in range(4):
            start_w = self.start_w_rate
            for j in range(6):
                self.roi_coordinate_rate.append([start_h, start_w])
                # print(start_w, start_h, start_w+square, start_h+square)
                start_w+=(square+padding)
            start_h+=(square+padding)
        # print('gen_roi_coordinate_rate', self.roi_coordinate_rate[0])

    def gen_RectItem(self):
        self.delete_all_item()
        w = self.img.shape[1]
        for coor in self.roi_coordinate_rate:
            item = GraphicItem(coor[1]*w, coor[0]*w, self.square_rate*w) # pixel座標
            item.setPos(0, 0)
            self._scene.addItem(item)
            # print(coor[0], coor[1])
        # print('gen_RectItem', self.roi_coordinate_rate[0])

class ROI_tune_window(QtWidgets.QWidget):
    to_main_window_signal = pyqtSignal(int , list)

    def __init__(self):
        super(ROI_tune_window, self).__init__()
        
        # Widgets
        self.viewer = ImageViewer(self)
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignCenter )
        self.label.setText('可使用滑鼠拖曳框框\n按下Ctrl可點選多個框或用滑鼠拉範圍遠取')
        self.btn_OK = QtWidgets.QPushButton(self)
        self.btn_OK.setText("OK")
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.label)
        VBlayout.addWidget(self.viewer)
        VBlayout.addWidget(self.btn_OK)

        # # 接受信號後要連接到什麼函數(將值傳到什麼函數)
        # self.viewer.mouse_release_signal.connect(self.get_roi_coordinate)
        self.btn_OK.clicked.connect(self.get_roi_coordinate)

        self.setStyleSheet(
                        "QWidget{background-color: rgb(66, 66, 66);}"
                        "QLabel{font-size:20pt; font-family:微軟正黑體; color:white;}"
                        "QPushButton{font-size:20pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}")

    def tune(self, tab_idx, img):
        self.tab_idx = tab_idx
        self.viewer.setPhoto(img)
        self.viewer.gen_RectItem()
        self.showMaximized()
    
    def get_roi_coordinate(self):
        items = self.viewer._scene.items()[:-1]
        items.reverse()
        roi_coordinate = []
        for item in items:
            scenePos = item.mapToScene(item.boundingRect().topLeft())
            r1, c1 = int(scenePos.y()), int(scenePos.x())
            scenePos = item.mapToScene(item.boundingRect().bottomRight())
            r2, c2 = int(scenePos.y()), int(scenePos.x())
            roi_coordinate.append([r1,c1,r2,c2])
            # print(r1,c1,r2,c2)
            
            # cv2.imshow('a', self.viewer.img[r1:r2,c1:c2,:])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # print()
        w = self.viewer.img.shape[1]
        self.viewer.roi_coordinate_rate = np.array(roi_coordinate)/w
        # for coor in self.viewer.roi_coordinate_rate:
            # print(coor[1], coor[0])
        # print('get_roi_coordinate', self.viewer.roi_coordinate_rate[0])
        self.to_main_window_signal.emit(self.tab_idx, roi_coordinate)
        self.close()





if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ROI_tune_window()
    filename = "CCM-Target.jpg"
    filename = "CCM-Target2.jpg"
    img = cv2.imdecode( np.fromfile( file = filename, dtype = np.uint8 ), cv2.IMREAD_COLOR )
    window.tune(img)
    sys.exit(app.exec_())