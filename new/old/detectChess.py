import cv2
import numpy as np

def color_detection(frame):
    # 定义颜色范围（在HSV颜色空间中）
    dst1 = cv2.GaussianBlur(frame, (9, 9), 0)
    dst2 = cv2.GaussianBlur(frame, (21, 21), 0)
    lower_white = np.array([102, 23, 210])
    upper_white = np.array([156, 142, 255])
    lower_black = np.array([73, 78, 5])
    upper_black = np.array([204, 255, 116])

    # 将帧转换为HSV颜色空间
    hsv_frame1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2HSV)
    hsv_frame2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2HSV)

    # 根据颜色范围创建掩膜
    white_mask = cv2.inRange(hsv_frame1, lower_white, upper_white)
    black_mask = cv2.inRange(hsv_frame2, lower_black, upper_black)


    # 对掩膜进行形态学操作，以去除噪声
    kernel1 = np.ones((15, 15), np.uint8)
    kernel2 = np.ones((9, 9), np.uint8)
    white_mask = cv2.dilate(white_mask, kernel2, iterations = 1)
    # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel1)
    cv2.imshow('white_inrange', white_mask)
    cv2.imshow('black_inrange', black_mask)

    # 在原始帧中找到颜色区域并绘制方框
    contours, _ = cv2.findContours(white_mask + black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        color = ""
        if cv2.contourArea(contour) > 600 and h * 1.2 > w and w * 1.2 > h:  # 设置最小区域面积以排除噪声
            if np.any(white_mask[y:y + h, x:x + w]):
                color = "white"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif np.any(black_mask[y:y + h, x:x + w]):
                color = "black"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(frame, color, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame



# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    # 进行颜色识别
    result = color_detection(frame)

    # 显示结果帧
    cv2.imshow("Color Detection", result)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()

# if __name__ == "__main__":
#     img = cv2.imread('5.png')
#     color_detection(img)
#     cv2.imshow('img', img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()