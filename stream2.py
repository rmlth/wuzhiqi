import platform
import threading
import cv2
import time
import numpy as np
from flask import Flask, render_template, Response
from loguru import logger
import fistquad_detector

# 创建四边形检测器对象
quad_detector = fistquad_detector.QuadDetector(
    max_perimeter=8000,
    min_perimeter=300,
    scale=1,
    min_angle=30,
    line_seg_num=3
)

class ThreadedCamera(object):
    """
    多线程摄像头类
    """
    def __init__(self, url=0, FPS=1/60):
        """
        :param url: 摄像头地址, 默认为0, 可以是0表示使用默认摄像头, 也可以是文件路径/视频路径
        """
        self.frame = None
        self.capture = cv2.VideoCapture(url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # 设置缓冲区大小

        self.FPS = FPS  # 设置摄像头帧率, 默认为60帧每秒
        self.FPS_MS = int(self.FPS * 1000)
    
        self.detection_times = []  # 检测时间记录

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """不断检查摄像头读取的帧"""
        while True:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if status:
                    self.frame = frame.copy()
            time.sleep(self.FPS)

    def process_frame(self, frame):
        """
        :param frame: 输入帧
        :return: 处理后的帧, 如果没有检测到四边形则返回原始帧
        """
        start_time = time.time()
    
        try:
            # 获取检测到的顶点、缩放顶点、交点和点列表
            vertices, scale_vertices, intersection, points_list = quad_detector.detect(frame)
            frame_drawed = quad_detector.draw()

            # 使用 generate_grid 生成网格
            grid_lines = quad_detector.generate_grid()

            # 使用 OpenCV 绘制网格
            for line in grid_lines:
                pt1, pt2 = tuple(line[0]), tuple(line[1])
                cv2.line(frame_drawed, pt1, pt2, (0, 255, 0), 1)  # 绘制网格

        except Exception as e:
            logger.error(f"检测过程中出现错误: {e}")
            frame_drawed = frame

        end_time = time.time()
        detection_time = end_time - start_time
        self.detection_times.append(detection_time)
        logger.info(f"检测时间: {detection_time}")

        return frame_drawed

    def show_frame(self):  
        cv2.namedWindow('Original MJPEG Stream', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Stream', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Edge Detection Stream', cv2.WINDOW_NORMAL)
        while True:
            frame = self.frame
            if frame is None or frame.size == 0:
                logger.error("摄像头没有帧")
                continue

            try:
                processed_frame = self.process_frame(frame)
                edge_frame = quad_detector.preprocess_image()

                if processed_frame is not None:
                    cv2.imshow('Processed Stream', processed_frame)
                cv2.imshow('Edge Detection Stream', quad_detector.pre_img)  
            except Exception as e:
                logger.error(f"处理帧时出错: {e}")

            cv2.imshow('Original MJPEG Stream', frame)
            key = cv2.waitKey(self.FPS_MS)
            if key == 27:  # 按 ESC 键
                break

        cv2.destroyAllWindows()


# Flask 服务器配置
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global url
    camera = ThreadedCamera(url)
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(camera):
    """
    生成发送到 Flask 服务器的帧
    """
    while True:
        frame = camera.frame
        if frame is not None:
            try:
                processed_frame = camera.process_frame(frame)
                if processed_frame is not None:
                    _, jpeg_buffer = cv2.imencode('.jpg', processed_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue

# 主程序入口
if __name__ == '__main__':
    # 摄像头URL设置为本地摄像头
    url = 2

    # 根据操作系统选择运行方式
    if platform.system() == 'Linux': 
        camera = ThreadedCamera(url)
        camera.show_frame()
    else:
        app.run(host='0.0.0.0', debug=True)