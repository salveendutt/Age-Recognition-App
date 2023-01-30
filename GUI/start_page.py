from PySide2.QtCore import (
    QSize,
    QPoint
)
from PySide2.QtWidgets import (
    QPushButton,
    QWidget
)


class StartPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.button_camfeed = QPushButton("Prediction on\ncamera feed")
        self.button_camfeed.setParent(self)
        self.button_camfeed.setFixedSize(QSize(200, 100))
        self.button_camfeed.move(QPoint(250, 300))
        self.button_camfeed.clicked.connect(self.camera_feed_page)
        self.button_camfeed.setStyleSheet("font-size: 20px;")

        self.button_image = QPushButton("Prediction on image")
        self.button_image.setParent(self)
        self.button_image.setFixedSize(QSize(200, 100))
        self.button_image.move(QPoint(550, 300))
        self.button_image.clicked.connect(self.image_page)
        self.button_image.setStyleSheet("font-size: 20px;")

    def camera_feed_page(self):
        self.parent().show_page(self.parent().camera_feed_page)

    def image_page(self):
        self.parent().show_page(self.parent().image_page)