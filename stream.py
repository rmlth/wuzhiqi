import platform
import threading
import cv2
import time
import numpy as np
from flask import Flask, render_template, Response
from loguru import logger
import quad_detector

quad_detector = quad_detector.QuadDetector(
    max_perimeter=8000,
    min_perimeter=500,
    scale=0.95,
    min_angle=10,
    line_seg_num=3
)

class ThreadedCamera:
    def __init__(self, url=2, FPS=1/30):
        self.capture = cv2.VideoCapture(url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.frame = None
        self.FPS = FPS
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame
            time.sleep(self.FPS)

    def process_frame(self, frame):
        try:
            vertices, scale_vertices, _, points_list = quad_detector.detect(frame)
            frame_drawed = quad_detector.draw()

            # 绘制3x3网格
            grid_lines = quad_detector.generate_3x3_grid()
            for line in grid_lines:
                pt1, pt2 = tuple(line[0]), tuple(line[1])
                cv2.line(frame_drawed, pt1, pt2, (0,255,0), 2)

            # 显示网格交点坐标
            for side in points_list:
                for pt in side[1:-1]:
                    cv2.putText(frame_drawed, f"{pt}", pt, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            
            return frame_drawed
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return frame

    def show_frame(self):
        cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
        while True:
            if self.frame is not None:
                processed = self.process_frame(self.frame)
                cv2.imshow('Processed', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = ThreadedCamera()
    cam.show_frame()