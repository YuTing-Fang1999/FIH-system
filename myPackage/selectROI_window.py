from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np

class ROI_coordinate(object):
    r1 = -1
    r2 = -1
    c1 = -1
    c2 = -1

class ImageViewer(QtWidgets.QGraphicsView):
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
        # 設置view可以進行鼠標的拖拽選擇
        self.setDragMode(self.RubberBandDrag)

        self.origin_pos = None
        self.roi_coordinate = ROI_coordinate()

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

    def setPhoto(self, pixmap=None):
        # self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())
        # self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def mousePressEvent(self, event):
        super(ImageViewer, self).mousePressEvent(event)
        if event.buttons() == Qt.LeftButton:
            self.origin_pos = event.pos()

            scenePos = self.mapToScene(event.pos()).toPoint()
            self.roi_coordinate.c1 = max(0,scenePos.x())
            self.roi_coordinate.r1 = max(0,scenePos.y())
    
    def mouseReleaseEvent(self, event):
        super(ImageViewer, self).mouseReleaseEvent(event)
        self.origin_pos = None

        scenePos = self.mapToScene(event.pos()).toPoint()
        self.roi_coordinate.c2 = min(self.img.shape[1], scenePos.x())
        self.roi_coordinate.r2 = min(self.img.shape[0], scenePos.y())

        self.set_ROI_draw()

    def set_ROI_draw(self):
        img = self.img.copy()
        # print(self.roi_coordinate.r1)

        if self.roi_coordinate.r1 != -1:
            coor = self.roi_coordinate
            cv2.rectangle(img, (coor.c1, coor.r1), (coor.c2, coor.r2), (0,0,255), 5)
            
        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*img.shape[2], QImage.Format_RGB888).rgbSwapped()
        self.setPhoto(QPixmap(qimg))

class SelectROI_window(QtWidgets.QWidget):
    to_main_window_signal = pyqtSignal(int, np.ndarray, ROI_coordinate)

    def __init__(self):
        super(SelectROI_window, self).__init__()
        self.filefolder = "./"

        # Widgets
        self.viewer = ImageViewer(self)
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignCenter )
        self.label.setText('按下Ctrl可以使用滑鼠縮放拖曳')
        self.btn_OK = QtWidgets.QPushButton(self)
        self.btn_OK.setText("OK")
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.label)
        VBlayout.addWidget(self.viewer)
        VBlayout.addWidget(self.btn_OK)

        # # 接受信號後要連接到什麼函數(將值傳到什麼函數)
        # self.viewer.mouse_release_signal.connect(self.get_roi_coordinate)
        self.btn_OK.clicked.connect(lambda:self.get_roi_coordinate(
            self.viewer.img, self.viewer.roi_coordinate
            )
        )

        self.setStyleSheet(
                        "QWidget{background-color: rgb(66, 66, 66);}"
                        "QLabel{font-size:20pt; font-family:微軟正黑體; color:white;}"
                        "QPushButton{font-size:20pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}")

    
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Control:
            print("press ctrl")
            self.viewer.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Control:
            print("release ctrl")
            self.viewer.setDragMode(self.viewer.RubberBandDrag)

    def open_img(self, tab_idx):
        self.tab_idx = tab_idx
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  self.filefolder, # start path
                  'Image Files(*.png *.jpg *.jpeg *.bmp)')    
        
        if filename == '': return
        self.filefolder = '/'.join(filename.split('/')[:-1])

        # filename = 'ColorChecker1.jpg'
        # load img
        img = cv2.imdecode( np.fromfile( file = filename, dtype = np.uint8 ), cv2.IMREAD_COLOR )
        self.viewer.img = img
        self.viewer.set_ROI_draw()

        self.showMaximized()
        self.show()

    def get_roi_coordinate(self, img, roi_coordinate):
        # roi_img = self.viewer.img[int(roi_coordinate.r1):int(roi_coordinate.r2), int(roi_coordinate.c1):int(roi_coordinate.c2)]
        # cv2.imshow('roi_img', roi_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.close()
        self.to_main_window_signal.emit(self.tab_idx, img, roi_coordinate)
        
    




if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SelectROI_window()
    window.open_img(0)
    sys.exit(app.exec_())