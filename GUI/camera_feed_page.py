import cv2
import qimage2ndarray

from PySide2.QtCore import (
    QPoint,
    QTimer,
    QSize
)
from PySide2.QtGui import (
    QPixmap
)
from PySide2.QtWidgets import (
    QPushButton,
    QWidget,
    QLabel
)

class CameraFeedPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_size = QSize(640, 480)

        self.image_label = QLabel()
        self.image_label.setParent(self)
        self.image_label.setFixedSize(self.video_size)
        self.image_label.move(QPoint(180, 20))
        self.image_label.setStyleSheet("border: 1px solid black; background-color: white;")

        self.button_back = QPushButton("Back page")
        self.button_back.setParent(self)
        self.button_back.setFixedSize(200, 100)
        self.button_back.move(QPoint(100, 510))
        self.button_back.clicked.connect(self.back_page)
        self.button_back.setStyleSheet("font-size: 20px;")

        self.button_start = QPushButton("Start camera")
        self.button_start.setParent(self)
        self.button_start.setFixedSize(200, 100)
        self.button_start.move(QPoint(400, 510))
        self.button_start.clicked.connect(self.start_camera)
        self.button_start.setStyleSheet("font-size: 20px;")

        self.button_stop = QPushButton("Stop camera")
        self.button_stop.setParent(self)
        self.button_stop.setFixedSize(200, 100)
        self.button_stop.move(QPoint(400, 630))
        self.button_stop.clicked.connect(self.stop_camera)
        self.button_stop.setStyleSheet("font-size: 20px;")

        self.button_predict = QPushButton("Start prediction")
        self.button_predict.setParent(self)
        self.button_predict.setFixedSize(200, 100)
        self.button_predict.move(QPoint(700, 510))
        self.button_predict.clicked.connect(self.predict)
        self.button_predict.setStyleSheet("font-size: 20px;")

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        self.capture.release()
        self.image_label.clear()

    def display_video_stream(self):
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = qimage2ndarray.array2qimage(frame)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def predict(self):
        # your prediction code goes to here
        pass

    def back_page(self):
        self.parent().show_page(self.parent().start_page)