import cv2
import numpy as np
import sys
from loguru import logger

class QuadDetector:
    def __init__(self, max_perimeter=9000, min_perimeter=300, scale=1, min_angle=30, line_seg_num=3):
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
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        self.pre_img = cv2.Canny(blur, 50, 150)

    def find_max_quad_vertices(self):
        contours, _ = cv2.findContours(self.pre_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_perimeter = 0
        
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) == 4:
                perimeter = cv2.arcLength(approx, True)
                if self.min_perimeter < perimeter < self.max_perimeter and perimeter > max_perimeter:
                    angles = []
                    for i in range(4):
                        pt1 = approx[i][0]
                        pt2 = approx[(i+1)%4][0]
                        pt3 = approx[(i+2)%4][0]
                        vec1 = pt1 - pt2
                        vec2 = pt3 - pt2
                        angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
                        angles.append(np.degrees(angle))
                    
                    if all(a >= self.min_angle for a in angles):
                        max_perimeter = perimeter
                        self.vertices = approx.reshape(4, 2)
        
        return self.vertices

    def find_scale_quad_vertices(self):
        def persp_trans(img, vertices):
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = vertices[0]
            rect[1] = vertices[3]
            rect[2] = vertices[2]
            rect[3] = vertices[1]

            height, width = img.shape[:2]
            dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            return M, np.linalg.inv(M)

        def shrink_rectangle(img, scale):
            h, w = img.shape[:2]
            center = (w//2, h//2)
            return np.array([
                [center[0] - int(w*scale/2), center[1] - int(h*scale/2)],
                [center[0] - int(w*scale/2), center[1] + int(h*scale/2)],
                [center[0] + int(w*scale/2), center[1] + int(h*scale/2)],
                [center[0] + int(w*scale/2), center[1] - int(h*scale/2)]
            ], dtype=np.int32)

        if self.vertices is not None:
            M, inv_M = persp_trans(self.img, self.vertices)
            small_rect = shrink_rectangle(self.img, self.scale)
            
            homog_coords = np.concatenate([small_rect.astype(np.float32), 
                                       np.ones((4,1))], axis=1)
            trans_coords = (inv_M @ homog_coords.T).T
            trans_coords = (trans_coords[:, :2] / trans_coords[:, 2:3]).astype(int)
            
            self.scale_vertices = trans_coords
            return self.scale_vertices

    def segment_line(self):
        def interpolate(p1, p2, n):
            return [tuple(np.round(p1 + (p2-p1)*i/n).astype(int)) for i in range(n+1)]
        
        if self.scale_vertices is not None:
            self.points_list = [
                interpolate(self.scale_vertices[0], self.scale_vertices[1], self.line_seg_num),
                interpolate(self.scale_vertices[1], self.scale_vertices[2], self.line_seg_num),
                interpolate(self.scale_vertices[2], self.scale_vertices[3], self.line_seg_num),
                interpolate(self.scale_vertices[3], self.scale_vertices[0], self.line_seg_num)
            ]
            return self.points_list

    def generate_3x3_grid(self):
        grid_lines = []
        # 横向网格线（左右边等分点连接）
        for i in range(1, 3):
            grid_lines.append([self.points_list[0][i], self.points_list[2][i]])
        # 纵向网格线（上下边等分点连接）
        for i in range(1, 3):
            grid_lines.append([self.points_list[1][i], self.points_list[3][i]])
        return grid_lines

    def draw(self):
        img_draw = self.img.copy()
        if self.vertices is not None:
            cv2.drawContours(img_draw, [self.vertices.astype(int)], -1, (0,0,255), 2)
            for pt in self.vertices:
                cv2.circle(img_draw, tuple(pt), 5, (255,0,0), -1)
        return img_draw

    def detect(self, img):
        self.img = img
        self.preprocess_image()
        self.find_max_quad_vertices()
        self.find_scale_quad_vertices()
        self.segment_line()
        return self.vertices, self.scale_vertices, None, self.points_list  # 修正缩进