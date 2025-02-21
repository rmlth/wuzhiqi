import cv2
import numpy as np
import sys
from loguru import logger

# 配置日志记录
logger.remove()
logger.add(sys.stderr, level="INFO")  

class QuadDetector:
    """
    四边形检测器类
    """
    def __init__(self, max_perimeter=99999, min_perimeter=10000, scale=0, min_angle=30, line_seg_num=4):
        """
        初始化四边形检测器
        :param max_perimeter: 最大周长
        :param min_perimeter: 最小周长
        :param scale: 缩放比例
        :param min_angle: 最小角度
        :param line_seg_num: 线段数量
        """
        self.img = None

        self.max_perimeter = max_perimeter
        self.min_perimeter = min_perimeter
        self.scale = scale
        self.min_angle = min_angle
        self.line_seg_num = line_seg_num

        self.vertices = None
        self.scale_vertices = None
        self.intersection = None
        self.points_list = None

    def preprocess_image(self):
        """
        预处理图像，包括灰度转换、高斯模糊和边缘检测
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        blur = cv2.GaussianBlur(gray, (1, 1), 0)  # 高斯模糊
        edges = cv2.Canny(blur, 50, 200)  # 边缘检测

        self.pre_img = edges

    def find_max_quad_vertices(self):
        """
        查找最大四边形的顶点
        """
        contours, _ = cv2.findContours(self.pre_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
        logger.debug(f'contours cnt: {len(contours)}')

        max_perimeter_now = 0

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.09 * cv2.arcLength(cnt, True), True)  # 近似多边形

            if len(approx) == 4:  # 检查是否为四边形
                perimeter = cv2.arcLength(approx, True)
                perimeter_allowed = (perimeter <= self.max_perimeter) and (perimeter >= self.min_perimeter)

                if perimeter_allowed and perimeter > max_perimeter_now:
                    cosines = []
                    for i in range(4):
                        p0 = approx[i][0]
                        p1 = approx[(i + 1) % 4][0]
                        p2 = approx[(i + 2) % 4][0]
                        v1 = p0 - p1
                        v2 = p2 - p1
                        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(cosine_angle) * 180 / np.pi
                        cosines.append(angle)

                    if all(angle >= self.min_angle for angle in cosines):
                        logger.info(f"perimeter: {perimeter}")
                        max_perimeter_now = perimeter
                        self.vertices = approx.reshape(4, 2)
                    else:
                        self.vertices = None

        logger.info(f"Found vertices: {self.vertices.tolist()}")
        return self.vertices
    
    def find_scale_quad_vertices(self):
        """
        查找缩放后的四边形顶点
        """
        def persp_trans(img, vertices):
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = vertices[0]
            rect[1] = vertices[3]
            rect[2] = vertices[2]
            rect[3] = vertices[1]

            height, width = img.shape[:2]
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            inv_M = np.linalg.inv(M)

            return M, inv_M
        
        def shrink_rectangle_new(img, scale):
            height, width = img.shape[:2]

            rectangle_vertices = [[0, 0], [0, height], [width, height], [width, 0]]

            center_x = width // 2
            center_y = height // 2

            small_vertices = []

            for vertex in rectangle_vertices:
                new_x = int(center_x + (vertex[0] - center_x) * scale)
                new_y = int(center_y + (vertex[1] - center_y) * scale)
                small_vertices.append([new_x, new_y])

            return np.array(small_vertices, dtype=np.int32)

        def inv_trans_vertices(small_vertices, inv_M):
            vertices_array = np.array(small_vertices, dtype=np.float32)
            vertices_homo = np.concatenate([vertices_array, np.ones((vertices_array.shape[0], 1))], axis=1)
            inv_trans_vertices_homo = np.dot(inv_M, vertices_homo.T).T
            inv_trans_vertices = inv_trans_vertices_homo[:, :2] / inv_trans_vertices_homo[:, 2, None]
            inv_trans_vertices_int = inv_trans_vertices.astype(int)

            return inv_trans_vertices_int
        
        _, inv_M = persp_trans(self.img, self.vertices)
        small_vertices = shrink_rectangle_new(self.img, self.scale)
        self.scale_vertices = inv_trans_vertices(small_vertices, inv_M)

        logger.debug(f"Found scale vertices: {self.scale_vertices}")
        return self.scale_vertices

    def segment_line(self, scale_vertices=None, line_seg_num=None):
        """
        根据缩放后的顶点和线段数量分割线段
        """
        if scale_vertices is None:
            scale_vertices = self.scale_vertices

        if line_seg_num is None:
            line_seg_num = self.line_seg_num  
    
        def average_points(point1, point2, N):
            delta_x = (point2[0] - point1[0]) / N
            delta_y = (point2[1] - point1[1]) / N
    
            points_list = []
    
            for i in range(N + 1):
                x = int(point1[0] + delta_x * i)
                y = int(point1[1] + delta_y * i)
                points_list.append([x, y])
    
            return points_list

        points_0 = average_points(scale_vertices[0], scale_vertices[1], line_seg_num)
        points_1 = average_points(scale_vertices[1], scale_vertices[2], line_seg_num)
        points_2 = average_points(scale_vertices[2], scale_vertices[3], line_seg_num)
        points_3 = average_points(scale_vertices[3], scale_vertices[0], line_seg_num)

        self.points_list = [points_0, points_1, points_2, points_3]
        logger.debug(f"Found points list: {self.points_list}")
        return self.points_list

    def generate_grid(self):
        """
        生成网格线
        """
        grid_lines = []
        for i in range(self.line_seg_num + 1):
            line = []
            line.append(self.points_list[0][i])
            line.append(self.points_list[2][i])
            grid_lines.append(line)

            line = []
            line.append(self.points_list[1][i])
            line.append(self.points_list[3][i])
            grid_lines.append(line)

        return grid_lines

    def calculate_intersection(self, vertices=None):
        """
        计算四边形对角线的交点
        :param vertices: 输入的顶点
        :return: 交点坐标
        """
        if vertices is None:
            vertices = self.vertices

        x1, y1 = vertices[0]
        x2, y2 = vertices[2]
        x3, y3 = vertices[1]
        x4, y4 = vertices[3]

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3

        det = dx1 * dy2 - dx2 * dy1

        if det == 0 or (dx1 == 0 and dx2 == 0) or (dy1 == 0 and dy2 == 0):
            return None
        dx3, dy3 = x1 - x3, y1 - y3
        det1 = dx1 * dy3 - dx3 * dy1
        det2 = dx2 * dy3 - dx3 * dy2

        if det1 == 0 or det2 == 0:
            return None

        s = det1 / det
        t = det2 / det

        if 0 <= s <= 1 and 0 <= t <= 1:
            intersection_x = int(x1 + dx1 * t)
            intersection_y = int(y1 + dy1 * t)
            self.intersection = [intersection_x, intersection_y]

            logger.info(f"Found intersection: {self.intersection}")
            return intersection_x, intersection_y
        else:
            logger.info(f"No intersection found.")
            return []

    def detect(self, img):
        """
        检测图像中的四边形
        :param img: 输入图像
        """
        self.img = img.copy()

        self.preprocess_image()
        self.vertices = self.find_max_quad_vertices()
        self.scale_vertices = self.find_scale_quad_vertices()
        self.intersection = self.calculate_intersection()
        self.points_list = self.segment_line()

        return self.vertices, self.scale_vertices, self.intersection, self.points_list
    
    def draw(self, img=None):
        """
        绘制检测结果
        :param img: 输入图像
        """
        if img is None:
            img = self.img.copy()

        def draw_point_text(img, x, y, bgr=(0, 0, 255)):
            cv2.circle(img, (x, y), 6, bgr, -1)
            cv2.putText(
                img,
                f"({x}, {y})",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )
            return img

        def draw_lines_points(img, vertices, bold=2):
            cv2.drawContours(img, [vertices], 0, (255, 0, 0), bold)

            for vertex in vertices:
                draw_point_text(img, vertex[0], vertex[1])
            
            cv2.line(
                img,
                (vertices[0][0], vertices[0][1]),
                (vertices[2][0], vertices[2][1]),
                (0, 255, 0), 1,
            )
            cv2.line(
                img,
                (vertices[1][0], vertices[1][1]),
                (vertices[3][0], vertices[3][1]),
                (0, 255, 0), 1,
            )
            return img
        
        def draw_segment_points(img, points_list):
            logger.debug(f"Found points list: {points_list}")
            for points in points_list:
                for point in points:
                    cv2.circle(img, (int(point[0]), int(point[1])), 4, (0, 255, 255), -1)
            
            return img

        img_drawed = draw_lines_points(self.img, self.vertices)          # 绘制四边形
        img_drawed = draw_lines_points(img_drawed, self.scale_vertices)  # 绘制缩放后的四边形
        img_drawed = draw_segment_points(img_drawed, self.points_list)   # 绘制分割点
        img_drawed = draw_point_text(img_drawed, self.intersection[0], self.intersection[1]) # 绘制交点

        return img_drawed

if __name__ == '__main__':
    print("初始化四边形检测器")
    cap = cv2.VideoCapture(0)

    quad_detector = QuadDetector()

    quad_detector.max_perimeter = 99999
    quad_detector.min_perimeter = 1
    quad_detector.scale = 500 / 600
    quad_detector.min_angle = 30
    quad_detector.line_seg_num = 6

    while True:
        ret, img = cap.read()
        if not ret:
            logger.error("无法读取摄像头帧")
            break

        vertices, scale_vertices, intersection, points_list = quad_detector.detect(img)
        img_detected = quad_detector.draw(img)

        cv2.imshow("检测结果", img_detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()