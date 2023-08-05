import numpy as np
import cv2 as cv

class Detector:
    def __init__(self) -> None:
        self.map_white = cv.imread('filter-loopylane.png')
        self.map_white_copy = self.map_white.copy()
        self.last_direction = np.array([-10,0])

    def main(self, filtered):
        pos, direction = self.detect_arrow(filtered)
        self.last_direction = direction
        pos_rounded, direction_rounded = self.round(pos, direction)
        minimap = self.draw_circle_on_map(pos_rounded, direction_rounded)
        return minimap
    
    def detect_arrow(self, filtered):
        def angle_between_vectors(v1, v2):
            dot_product = np.dot(v1, v2)
            magnitude_v1 = np.linalg.norm(v1)
            magnitude_v2 = np.linalg.norm(v2)
            angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
            return angle_rad
        
        arrow_pixels = np.argwhere(filtered == 255)
        mean_pixel = np.mean(arrow_pixels, axis=0)
        centered_data = arrow_pixels - mean_pixel

        _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
        direction_estimate_raw = Vt[0][::-1]
        direction_estimate = (direction_estimate_raw / np.linalg.norm(direction_estimate_raw)) * 10

        print(f"Last direction {self.last_direction}\nCurrent direction {direction_estimate}")
        # if angle_between_vectors(self.last_direction, direction_estimate) > np.pi/2:
        #     print("Switch!")
        #     direction_estimate = -direction_estimate

        if np.linalg.norm(self.last_direction + direction_estimate) < 10:
            print("Switch!")
            direction_estimate = -direction_estimate

        pos_estimate = mean_pixel[::-1]

        return pos_estimate, direction_estimate
    
    def round(self, pos, direction):
        return np.round(pos).astype(int), np.round(direction).astype(int)
    
    def draw_circle_on_map(self, pos, direction):
        self.map_white = self.map_white_copy.copy()

        radius, color, thickness = 3, (0,255,0), -1

        cv.circle(self.map_white, pos, radius, color, thickness)
        cv.arrowedLine(self.map_white,pos, pos+direction,(255,0,0),3)

        print(f"Direction {direction}")
        print(f"Current position {pos}")

        return self.map_white