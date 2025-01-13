import cv2
import numpy as np

def color_detection(frame):
    # 转换到HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义更精确的颜色范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # 创建掩膜
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    black_mask = cv2.inRange(hsv_frame, lower_black, upper_black)

    # 形态学处理以减少噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    # 轮廓检测
    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制白棋轮廓
    for contour in contours_white:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'White', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 绘制黑棋轮廓
    for contour in contours_black:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Black', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# 打开摄像头
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = color_detection(frame)
    cv2.imshow("Color Detection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
