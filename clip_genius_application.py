import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon

def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(500, 250, 500, 500)
    win.setWindowTitle("Clip Genius")
    win.setWindowIcon(QIcon("hero-img.png"))
    win.show()
    sys.exit(app.exec_())
    

window()

#python clip_genius_application.py
