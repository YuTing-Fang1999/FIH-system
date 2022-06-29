from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import cv2
from .ROI import ROI
import numpy as np

class ImageViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, tab_idx, parent = None):
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

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # super().keyPressEvent(event)
        if event.key() == Qt.Key_Control:
            print("press ctrl")
            self.mouseDrag()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        # super().keyPressEvent(event)
        if event.key() == Qt.Key_Control:
            print("release ctrl")
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
        self.deleteROI()
        self.fitInView()

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            # self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            # self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()
    
    def toggleDragMode(self):
        print('toggleDragMode')
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mouseDragWheelEvent(self, event):
        self.deleteROI()
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
        # super(ImageViewer, self).wheelEvent(event)

    def deleteROI(self):
        self.rubberBand.hide()
        self.ROI.img_roi = None
        self.ROI.x1 = None
        self.ROI.x2 = None
        self.ROI.y1 = None
        self.ROI.y2 = None

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
        # print(f"[show_mouse_press] {event.x()=}, {event.y()=}, {event.button()=}")

        # if self._photo.isUnderMouse():
        #     self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())

        self.origin_pos = event.pos()
        scenePos = self.mapToScene(event.pos()).toPoint()
        self.ROI.set_x1_y1(scenePos.x(), scenePos.y())
        self.ROI.img_roi = None
    
        self.rubberBand.setGeometry(QtCore.QRect(self.origin_pos, QtCore.QSize()))  # QSize() 此時爲-1 -1
        self.rubberBand.show()
        cv2.destroyAllWindows()

    def mouseMoveRubberBand(self, event):
        # print(f"[show_mouse_move] {event.x()=}, {event.y()=}, {event.button()=}")
        # print(event.pos())
        if self.origin_pos:
            self.rubberBand.setGeometry(QtCore.QRect(self.origin_pos, event.pos()).normalized())  # 這裏可以

    def mouseReleaseRubberBand(self, event):
        # print(f"[show_mouse_release] {event.x()=}, {event.y()=}, {event.button()=}")
        if not isinstance(self.ROI.img, np.ndarray):
            self.rubberBand.hide()
            return

        scenePos = self.mapToScene(event.pos()).toPoint()
        self.ROI.set_x2_y2(scenePos.x(), scenePos.y())
        img_roi = self.ROI.get_ROI()

        if img_roi is None: 
            self.deleteROI()
        else: 
            cv2.destroyAllWindows()
            img = img_roi.copy()
            self.ROI.draw_24_block_img = self.draw_24_block(img)
            cv2.imshow('roi '+str(self.tab_idx), self.ROI.draw_24_block_img)
            cv2.waitKey(100)

            # self.default_ROI = [ROI.x1, ROI.y1, ROI.x2, ROI.y2]
        self.origin_pos = None

    def draw_24_block(self, img):
        h, w, c = img.shape
        color = (0, 0, 255) # red
        thickness = img.shape[1]//200 # 寬度 (-1 表示填滿)
        square = 0.08*w
        padding = 0.088*w

        start_h = 0.038*w
        for i in range(4):
            start_w = 0.038*w
            for j in range(6):
                cv2.rectangle(img, (int(start_w), int(start_h)), (int(start_w+square), int(start_h+square)), color, thickness)
                start_w+=(square+padding)
            start_h+=(square+padding)
        return self.ResizeWithAspectRatio(img, width = 400)

    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)
        


class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = ImageViewer(self)
        # 'Load image' button
        self.btnLoad = QtWidgets.QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
        # Button to change from drag/pan to getting pixel info
        self.btnPixInfo = QtWidgets.QToolButton(self)
        self.btnPixInfo.setText('Enter pixel info mode')
        self.btnPixInfo.clicked.connect(self.pixInfo)
        self.editPixInfo = QtWidgets.QLineEdit(self)
        self.editPixInfo.setReadOnly(True)
        self.viewer.photoClicked.connect(self.photoClicked)
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnPixInfo)
        HBlayout.addWidget(self.editPixInfo)
        VBlayout.addLayout(HBlayout)

    def loadImage(self):
        self.viewer.setPhoto(QtGui.QPixmap('ColorChecker1.jpg'))

    def pixInfo(self):
        self.viewer.toggleDragMode()

    def photoClicked(self, pos):
        if self.viewer.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec_())