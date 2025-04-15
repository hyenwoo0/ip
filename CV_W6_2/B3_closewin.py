## Basic-3. 창 닫기.

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon


class MyApp(QWidget):

  def __init__(self):
      super().__init__()
      self.initUI()

  def initUI(self):
      self.setWindowTitle('Icon')
    #   self.setWindowIcon(QIcon('C:/Users/chaew/source/DL_App/W11/web.png')) # Add app icon
      self.setWindowIcon(QIcon('web.png')) # Add app icon

      btn = QPushButton('Quit', self)
      btn.move(50, 50)
      btn.resize(btn.sizeHint())
      btn.clicked.connect(QCoreApplication.instance().quit)

      self.setWindowTitle('Quit Button')
      self.setGeometry(300, 300, 300, 200)
      self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
