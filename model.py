import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def calculate_area(tl, tr, br, bl):
    width = np.linalg.norm(tr - tl)
    height = np.linalg.norm(bl - tl)
    return width * height / 729

def compute_grid_centers(pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width, height = 300, 300
    dst_rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst_rect)
    grid_centers = [(int((j + 0.5) * (width / 3)), int((i + 0.5) * (height / 3))) for i in range(3) for j in range(3)]
    inv_M = np.linalg.inv(M)
    original_centers = []
    for (cx, cy) in grid_centers:
        original_pt = np.dot(inv_M, [cx, cy, 1])
        original_pt /= original_pt[2]
        original_centers.append((int(original_pt[0]), int(original_pt[1])))
    return original_centers

def get_color_category(image, centers):
    categories = []
    for (cx, cy) in centers:
        if 0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]:
            color = image[cy, cx]
            gray = int(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
            categories.append(0 if gray > 200 else 1 if gray < 50 else 2)
        else:
            categories.append(2)
    return categories

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise IOError("无法打开摄像头。")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头帧失败。")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area, largest_contour = 0, None

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                approx = np.squeeze(approx)
                area = calculate_area(*order_points(approx))
                if area > largest_area:
                    largest_area, largest_contour = area, approx

        if largest_contour is not None:
            try:
                grid_centers = compute_grid_centers(largest_contour)
                categories = get_color_category(frame, grid_centers)
                print("九个矩形框的中心点坐标（按行列顺序排序）：")
                for i, (cx, cy) in enumerate(grid_centers):
                    print(f"矩形 {i + 1}: ({cx}, {cy})")
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"({cx}, {cy})", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except ValueError as ve:
                print(f"错误: {ve}")

        cv2.imshow('Detected Chessboard Grid', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
