import numpy as np
import cv2 as cv

class Detector:
    def __init__(self) -> None:
        self.map = cv.imread('loopy-lane-map.png')
        self.map_white = cv.imread('filter-loopylane.png')
        self.map_white_copy = self.map_white.copy()

        self.frame_count = 40
        self.last_frames = np.array([])

    def main(self, filtered):
        pos, direction = self.detect_arrow(filtered)
        pos_rounded, direction_rounded = self.round(pos, direction)
        minimap = self.draw_circle_on_map(pos_rounded, direction_rounded)
        return minimap
    
    def detect_arrow(self, filtered):
        arrow_pixels = np.argwhere(filtered == 255)
        mean_pixel = np.mean(arrow_pixels, axis=0)

        centered_data = arrow_pixels - mean_pixel

        _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)

        direction_estimate_raw = Vt[0][::-1]
        direction_estimate = (direction_estimate_raw / np.linalg.norm(direction_estimate_raw)) * 10

        pos_estimate = mean_pixel[::-1]

        return pos_estimate, direction_estimate
    
    def round(self, pos, direction):
        return np.round(pos).astype(int), np.round(direction).astype(int)
    
    def draw_circle_on_map(self, pos, direction):
        self.map_white = self.map_white_copy.copy()
        self.map_white = self.map_white_copy.copy()
        # Draw a circle at the specified position
        radius = 3
        color = (0, 255, 0)  # Green color (in BGR format)
        thickness = -1  # Fill the circle (-1), set to a positive value for a circle outline
        cv.circle(self.map_white, pos, radius, color, thickness)
        
        print(f"Direction {direction}")
        print(f"Current position {pos}")
        
        cv.arrowedLine(self.map_white,pos, pos+direction,(255,0,0),3)

        return self.map_white