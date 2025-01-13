import cv2
import numpy as np

def detect_chessboard_from_camera():
    # 打开摄像头
    cap = cv2.VideoCapture(2)  # 参数0表示默认摄像头

    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        # 使用边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 使用霍夫直线变换检测棋盘线条
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 检测棋子 (假设棋子为圆形)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=50, param2=30, minRadius=10, maxRadius=30)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # 圆心
                radius = i[2]          # 半径
                cv2.circle(frame, center, radius, (0, 0, 255), 3)

        # 显示检测结果
        cv2.imshow("Chessboard Detection", frame)

        # 按 'q' 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

# 调用摄像头检测函数
detect_chessboard_from_camera()
