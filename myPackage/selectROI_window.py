from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import cv2
from ROI import ROI
import numpy as np

class ImageViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, tab_idx, function = "", parent = None):
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

        self.origin_pos = None
        self.tab_idx = tab_idx
        self.rubberBand = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self.ROI = ROI(self, self.rubberBand)
        self.mouseRubberBand()

    def hasPhoto(self):
        return not self._empty

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

    def resizeEvent(self, event):
        self.fitInView()

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def mouseDragWheelEvent(self, event):
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

    def mouseDrag(self):
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.mousePressEvent = lambda event: super(ImageViewer, self).mousePressEvent(event)
        self.mouseMoveEvent = lambda event: super(ImageViewer, self).mouseMoveEvent(event)
        self.mouseReleaseEvent = lambda event: super(ImageViewer, self).mouseReleaseEvent(event)
        self.wheelEvent = lambda event: self.mouseDragWheelEvent(event)
    
    def mouseRubberBand(self):
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.mousePressEvent = lambda event: self.mousePressRubberBand(event)
        self.mouseMoveEvent = lambda event: self.mouseMoveRubberBand(event)
        self.mouseReleaseEvent = lambda event: self.mouseReleaseRubberBand(event)
        self.wheelEvent = lambda event: None

    def mousePressRubberBand(self, event):
        self.origin_pos = event.pos()
        scenePos = self.mapToScene(event.pos()).toPoint()
        self.ROI.set_x1_y1(scenePos.x(), scenePos.y())
        self.ROI.roi_img = None
    
        self.rubberBand.setGeometry(QtCore.QRect(self.origin_pos, QtCore.QSize()))  # QSize() 此時爲-1 -1
        self.rubberBand.show()
        cv2.destroyAllWindows()

    def mouseMoveRubberBand(self, event):
        # print(f"[show_mouse_move] {event.x()=}, {event.y()=}, {event.button()=}")
        # print(event.pos())
        if self.origin_pos: self.rubberBand.setGeometry(QtCore.QRect(self.origin_pos, event.pos()).normalized())  # 這裏可以

    def mouseReleaseRubberBand(self, event):
        # print(f"[show_mouse_release] {event.x()=}, {event.y()=}, {event.button()=}")
        self.rubberBand.hide()

        scenePos = self.mapToScene(event.pos()).toPoint()
        self.ROI.set_x2_y2(scenePos.x(), scenePos.y())
        self.ROI.set_ROI()

        self.origin_pos = None


class SelectROI_window(QtWidgets.QWidget):
    def __init__(self, tabidx):
        super(SelectROI_window, self).__init__()
        self.viewer = ImageViewer(self)
        # self.viewer.ROI.set_img(img, self.viewer)
        # 'Load image' button
        self.btnReset = QtWidgets.QPushButton(self)
        self.btnReset.setText('Reset')
        # self.btnReset.clicked.connect()
        # Button to change from drag/pan to getting pixel info
        self.btnOk = QtWidgets.QPushButton(self)
        self.btnOk.setText('OK')
        # self.btnOk.clicked.connect()
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        # HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.btnReset)
        HBlayout.addWidget(self.btnOk)
        VBlayout.addLayout(HBlayout)

        self.setStyleSheet(
                        "QWidget{background-color: rgb(66, 66, 66);}"
                        "QLabel{font-size:20pt; font-family:微軟正黑體;}"
                        "QPushButton{font-size:20pt; font-family:微軟正黑體; background-color:rgb(255, 170, 0);}")

    
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Control:
            print("press ctrl")
            self.viewer.mouseDrag()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Control:
            print("release ctrl")
            self.viewer.mouseRubberBand()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    filename = 'ColorChecker1.jpg'
    # load img
    img = cv2.imdecode( np.fromfile( file = filename, dtype = np.uint8 ), cv2.IMREAD_COLOR )
    window = SelectROI_window(img)
    window.showMaximized()
    window.show()
    sys.exit(app.exec_())