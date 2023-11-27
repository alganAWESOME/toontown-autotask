import numpy as np
import math
import cv2 as cv

class Utils:
    @staticmethod
    def manhattan_dist(p1, p2):
        return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])
    
    @staticmethod
    def euclidean_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calc_centroid(contour):
        M = cv.moments(contour)
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])