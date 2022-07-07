# https://stackoverflow.com/questions/64290561/qlabel-correct-positioning-for-text-outline

import sys, math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
        
class OutlinedLabel(QLabel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = 1 / 25
        self.mode = True
        self.setBrush(Qt.white)
        self.setPen(Qt.black)

    def scaledOutlineMode(self):
        return self.mode

    def setScaledOutlineMode(self, state):
        self.mode = state

    def outlineThickness(self):
        return self.w * self.font().pointSize() if self.mode else self.w

    def setOutlineThickness(self, value):
        self.w = value

    def setBrush(self, brush):
        if not isinstance(brush, QBrush):
            brush = QBrush(brush)
        self.brush = brush

    def setPen(self, pen):
        if not isinstance(pen, QPen):
            pen = QPen(pen)
        pen.setJoinStyle(Qt.RoundJoin)
        self.pen = pen

    def sizeHint(self):
        w = math.ceil(self.outlineThickness() * 2)
        return super().sizeHint() + QSize(w, w)
    
    def minimumSizeHint(self):
        w = math.ceil(self.outlineThickness() * 2)
        return super().minimumSizeHint() + QSize(w, w)
    
    def paintEvent(self, event):
        w = self.outlineThickness()
        rect = self.rect()
        metrics = QFontMetrics(self.font())
        tr = metrics.boundingRect(self.text()).adjusted(0, 0, int(w), int(w))
        if self.indent() == -1:
            if self.frameWidth():
                indent = (metrics.boundingRect('x').width() + w * 2) / 2
            else:
                indent = w
        else:
            indent = self.indent()

        if self.alignment() & Qt.AlignLeft:
            x = rect.left() + indent - min(metrics.leftBearing(self.text()[0]), 0)
        elif self.alignment() & Qt.AlignRight:
            x = rect.x() + rect.width() - indent - tr.width()
        else:
            x = (rect.width() - tr.width()) / 2
            
        if self.alignment() & Qt.AlignTop:
            y = rect.top() + indent + metrics.ascent()
        elif self.alignment() & Qt.AlignBottom:
            y = rect.y() + rect.height() - indent - metrics.descent()
        else:
            y = (rect.height() + metrics.ascent() - metrics.descent()) / 2

        path = QPainterPath()
        path.addText(x, y, self.font(), self.text())
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        self.pen.setWidthF(w * 2)
        qp.strokePath(path, self.pen)
        if 1 < self.brush.style() < 15:
            qp.fillPath(path, self.palette().window())
        qp.fillPath(path, self.brush)


class Template(QWidget):

    def __init__(self):
        super().__init__()
        vbox = QVBoxLayout(self)
        label = OutlinedLabel('Lorem ipsum dolor sit amet consectetur adipiscing elit,')
        label.setStyleSheet('font-family: Monaco; font-size: 20pt')
        vbox.addWidget(label)

        label = OutlinedLabel('sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.')
        label.setStyleSheet('font-family: Helvetica; font-size: 30pt; font-weight: bold')
        vbox.addWidget(label)

        label = OutlinedLabel('Ut enim ad minim veniam,', alignment=Qt.AlignCenter)
        label.setStyleSheet('font-family: Comic Sans MS; font-size: 40pt')
        vbox.addWidget(label)

        label = OutlinedLabel('quis nostrud exercitation ullamco laboris nisi ut')
        label.setStyleSheet('font-family: Arial; font-size: 50pt; font-style: italic')
        vbox.addWidget(label)

        label = OutlinedLabel('aliquip ex ea commodo consequat.')
        label.setStyleSheet('font-family: American Typewriter; font-size: 60pt')
        label.setPen(Qt.red)
        vbox.addWidget(label)

        label = OutlinedLabel('Duis aute irure dolor', alignment=Qt.AlignRight)
        label.setStyleSheet('font-family: Luminari; font-size: 70pt')
        label.setPen(Qt.red); label.setBrush(Qt.black)
        vbox.addWidget(label)

        label = OutlinedLabel('in reprehenderit')
        label.setStyleSheet('font-family: Zapfino; font-size: 80pt')
        label.setBrush(Qt.red)
        vbox.addWidget(label)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Template()
    window.show()
    sys.exit(app.exec_())
