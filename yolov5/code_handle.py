import math
from time import time
import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from yolov5.tracker import CentroidTracker

tracker = CentroidTracker()


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(700, 500, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)


class live_stream(QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(live_stream, self).__init__()
        self.center_points = {}
        self.id_count = 1
        self.old_objects_ids = None

    def run(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()  # load model
        self.classes = self.model.names
        self.out_file = "Labeled_Video.avi"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.run_program()

    def get_video_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        return cv2.VideoCapture("q.mp4")  #q.mp4

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/yolov5s.pt')  # ultralytics/yolov5
        model = torch.hub.load('', 'custom', path='yolov5s.pt', source='local')  # local repo
        model.classes = 2
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param rects: for rectangle copy after predict
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        rects = []
        detections = []  # list thêm đối tượng
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                detections.append([x1, y1, x2 - x1, y2 - y1])
                rects.append([x1, y1, x2, y2])
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]) + " " + str(round(row[4], 2)), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        rects_ids = tracker.update(rects)
        for objectID, centroid in rects_ids.items():
            cv2.putText(frame, str(objectID), centroid, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
            print("show rects: ", rects_ids)
        return frame

    def run_program(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player = self.get_video_from_url()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time()
            ret, frame = player.read()
            assert ret
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / (np.round(end_time - start_time, 3))
            print(f"Frames Per Second : {round(fps, 2)} FPS")
            # out.write(frame)
            self.signal.emit(frame)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()
