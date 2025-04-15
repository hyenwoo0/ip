## Ex 3-6. 메뉴바 만들기.

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp
from PyQt5.QtGui import QIcon


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Add app icon
        self.setWindowTitle('Icon')
        # self.setWindowIcon(QIcon('C:/Users/chaew/source/DL_App/W11/web.png')) 
        self.setWindowIcon(QIcon('web.png')) 

        # Add menu 'Exit'
        # exitAction = QAction(QIcon('C:/Users/chaew/source/DL_App/W11/exit.png'), 'Exit', self)
        exitAction = QAction(QIcon('exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        # Add status bar
        self.statusBar()

        # menubar에 exit 메뉴 연결하기
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(exitAction)

        self.setWindowTitle('Menubar')
        self.setGeometry(300, 300, 300, 200)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)    # create pyqt5 app
    ex = MyApp()                    # create the instance of MyApp
    sys.exit(app.exec_())           # start the pyqt5 app
