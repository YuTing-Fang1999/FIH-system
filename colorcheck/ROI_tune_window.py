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

        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*img.shape[2], QImage.Format_RGB888).rgbSwapped()
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


    def gen_RectItem(self):
        self.delete_all_item()
        if self.roi_coordinate == None:
            self.roi_coordinate = []
            h, w, c = self.img.shape
            square = self.square_rate*w
            padding = self.padding_rate*w

            start_h = self.start_h_rate*w
            for i in range(4):
                start_w = self.start_w_rate*w
                for j in range(6):
                    self.roi_coordinate.append([start_w, start_h, square])
                    # print(start_w, start_h, start_w+square, start_h+square)
                    start_w+=(square+padding)
                start_h+=(square+padding)

        for coor in self.roi_coordinate:
            item = GraphicItem(coor[0], coor[1], coor[2])
            self._scene.addItem(item)

class SelectROI_window(QtWidgets.QWidget):
    def __init__(self):
        super(SelectROI_window, self).__init__()
        
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

        img = cv2.imdecode( np.fromfile( file = 'CCM-Target.jpg', dtype = np.uint8 ), cv2.IMREAD_COLOR )
        self.viewer.setPhoto(img)
        self.viewer.gen_RectItem()

        # # 接受信號後要連接到什麼函數(將值傳到什麼函數)
        # self.viewer.mouse_release_signal.connect(self.get_roi_coordinate)
        self.btn_OK.clicked.connect(lambda:self.get_roi_coordinate(
                self.viewer._scene.items()
            )
        )

        self.setStyleSheet(
                        "QWidget{background-color: rgb(66, 66, 66);}"
                        "QLabel{font-size:20pt; font-family:微軟正黑體; color:white;}"
                        "QPushButton{font-size:20pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}")

    def tune(self, img):
        self.viewer.setPhoto(img)
        self.viewer.gen_RectItem()
    
    def get_roi_coordinate(self, items):
        items.reverse()
        print(self.viewer.img.shape)
        self.roi_coordinate = []
        for item in items[1:]:
            scenePos = item.boundingRect().topLeft()
            r1, c1 = int(scenePos.y()), int(scenePos.x())
            scenePos = item.boundingRect().bottomRight()
            r2, c2 = int(scenePos.y()), int(scenePos.x())
            self.roi_coordinate = [r1,c1,r2,c2]
            # print(r1,c1,r2,c2)
            # cv2.imshow('a', self.viewer.img[r1:r2,c1:c2,:])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        self.close()
    




if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SelectROI_window()
    window.showMaximized()
    window.show()
    sys.exit(app.exec_())