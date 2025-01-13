import cv2
import numpy as np

def detect_chessboard(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 使用霍夫直线变换检测棋盘线条
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # 在原图上绘制线条
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 检测棋子 (假设棋子是圆形)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=20)

    # 在原图上绘制检测到的棋子
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])  # 圆心
            radius = i[2]          # 半径
            cv2.circle(image, center, radius, (0, 0, 255), 3)

    # 显示结果
    cv2.imshow("Chessboard Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用函数
detect_chessboard('/home/fishros/桌面/wuzhiqi/image2.jpg')
