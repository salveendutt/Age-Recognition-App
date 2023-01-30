from PySide2.QtCore import (
    QSize,
    Qt,
    QPoint
)
from PySide2.QtGui import (
    QPixmap
)
from PySide2.QtWidgets import (
    QPushButton,
    QLabel,
    QWidget,
    QFileDialog
)

class ImagePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_label = QLabel(self)
        self.image_label.setParent(self)
        self.image_label.setFixedSize(QSize(960, 600))
        self.image_label.move(QPoint(20, 20))
        self.image_label.setStyleSheet("border: 1px solid black; background-color: white;")

        self.button_back = QPushButton("Back page")
        self.button_back.setParent(self)
        self.button_back.setFixedSize(200, 100)
        self.button_back.move(QPoint(100, 640))
        self.button_back.clicked.connect(self.back_page)
        self.button_back.setStyleSheet("font-size: 20px;")

        self.button_load = QPushButton("Load Image")
        self.button_load.setParent(self)
        self.button_load.setFixedSize(200, 100)
        self.button_load.move(QPoint(400, 640))
        self.button_load.clicked.connect(self.load_image)
        self.button_load.setStyleSheet("font-size: 20px;")

        self.button_predict = QPushButton("Predict")
        self.button_predict.setParent(self)
        self.button_predict.setFixedSize(200, 100)
        self.button_predict.move(QPoint(700, 640))
        self.button_predict.clicked.connect(self.predict)
        self.button_predict.setStyleSheet("font-size: 20px;")

    def load_image(self):
        filters = "PNG File (*.png);;JPEG File (*.jpeg);;JPG File (*.jpg)"
        self.filename, filter = QFileDialog.getOpenFileName(self, filter=filters)
        QPixmap(self.filename).scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap(self.filename).scaled(self.image_label.size(), Qt.KeepAspectRatio))
        self.image_label.setAlignment(Qt.AlignCenter)

    def predict(self):
        # your prediction code goes to here
        pass

    def back_page(self):
        self.parent().show_page(self.parent().start_page)