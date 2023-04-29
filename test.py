import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget

class Page(QWidget):
    def __init__(self):
        super().__init__()
        # Create three buttons
        self.button1 = QPushButton("Button 1")
        self.button2 = QPushButton("Button 2")
        self.button3 = QPushButton("Button 3")

        # Connect the buttons to their respective functions
        # self.button1.clicked.connect(lambda: self.showSecondUI(self.button1))
        # self.button2.clicked.connect(lambda: self.showSecondUI(self.button2))
        # self.button3.clicked.connect(lambda: self.showSecondUI(self.button3))

        # Add the buttons to a vertical layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.button1)
        vbox.addWidget(self.button2)
        vbox.addWidget(self.button3)

        # Create a widget to hold the layout
        self.setLayout(vbox)
        
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.page1 = Page()
        self.setCentralWidget(self.page1)
        

    def showSecondUI(self, button):
        # Create two buttons and a return button
        button1 = QPushButton("Button 1")
        button2 = QPushButton("Button 2")
        returnButton = QPushButton("Return")

        # Connect the buttons to their respective functions
        button1.clicked.connect(self.doSomething)
        button2.clicked.connect(self.doSomething)

        # Add the buttons to a horizontal layout
        hbox = QHBoxLayout()
        hbox.addWidget(button1)
        hbox.addWidget(button2)

        # Add the layout and return button to a vertical layout
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(returnButton)

        # Create a widget to hold the layout
        widget = QWidget()
        widget.setLayout(vbox)

        # Set the widget as the central widget of the main window
        self.setCentralWidget(widget)

        # Connect the return button to the showPreviousPage function
        returnButton.clicked.connect(lambda: self.showPreviousPage())

    def showPreviousPage(self):
        # Set the previous page as the central widget
        self.setCentralWidget(self.previousPage)

    def doSomething(self):
        # Do something when a button in the second UI is clicked
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
