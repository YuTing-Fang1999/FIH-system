# https://blog.csdn.net/qq_25000387/article/details/106025439

import sys
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QMainWindow

from PyQt5.QtWidgets import QGraphicsItem, QGraphicsRectItem
from PyQt5.QtGui import QPixmap

class GraphicItem(QGraphicsRectItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPen(Qt.red)
        r = QRectF(0, 0, 50, 50) #起始座標,長,寬
        self.setRect(r)
        self.setFlag(QGraphicsItem.ItemIsSelectable)  # ***设置图元是可以被选择的
        self.setFlag(QGraphicsItem.ItemIsMovable)     # ***设置图元是可以被移动的


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self)
        # 有view就要有scene
        self.view.setScene(self.scene)
        # 设置view可以进行鼠标的拖拽选择
        self.view.setDragMode(self.view.RubberBandDrag)

        

        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.setCentralWidget(self.view)
        self.setWindowTitle("Graphics Demo")

    #override
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_N:
        # 当按下N键时，会在scene的（0,0）位置出现此图元
            item = GraphicItem()
            # item.setPos(0, 0)
            self.scene.addItem(item)

        if event.key() == Qt.Key_S:
            for item in self.scene.items()[:1]:
                scenePos = item.mapToScene(item.boundingRect().topLeft())
                print(scenePos.x()+0.5, scenePos.y()+0.5)
                scenePos = item.mapToScene(item.boundingRect().bottomRight())
                print(scenePos.x()-0.5, scenePos.y()-0.5)


def demo_run():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    demo_run()

