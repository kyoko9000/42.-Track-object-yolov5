import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from code_handle import live_stream, convert_cv_qt
from gui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.uic.Button_start.clicked.connect(self.start_capture_video)
        self.uic.Button_stop.clicked.connect(self.stop_capture_video)

        self.thread = {}

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.thread[1].stop()
        self.close()

    def start_capture_video(self):
        self.thread[1] = live_stream(index=1)
        self.thread[1].start()

        self.thread[1].signal.connect(self.show_wedcam)

    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = convert_cv_qt(cv_img)
        self.uic.label.setPixmap(qt_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())