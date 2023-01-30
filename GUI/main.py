import sys
import start_page
import camera_feed_page
import image_page

from PySide2.QtCore import (
    QSize
)
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Application")
        self.setFixedSize(QSize(1000, 800))

        self.start_page = start_page.StartPage(self)
        self.camera_feed_page = camera_feed_page.CameraFeedPage(self)
        self.image_page = image_page.ImagePage(self)

        self.show_page(self.start_page)

    def show_page(self, page):
        if self.centralWidget() is not None:
            self.centralWidget().setParent(None)
        self.setCentralWidget(page)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())