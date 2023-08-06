import numpy as np
import cv2 as cv

class Detector:
    def __init__(self) -> None:
        self.map_white = cv.imread('filter-loopylane.png')
        self.last_direction = np.array([-10,0])
        self.nodes = self.create_nodes()
        print(self.nodes)
        #self.draw_nodes()
        self.map_white_copy = self.map_white.copy()

    def main(self, filtered):
        pos, direction = self.detect_arrow(filtered)
        self.last_direction = direction
        pos_rounded, direction_rounded = self.round(pos, direction)
        minimap = self.visualize(pos_rounded, direction_rounded)
        return minimap
    
    def detect_arrow(self, filtered):
        arrow_pixels = np.argwhere(filtered == 255)
        mean_pixel = np.mean(arrow_pixels, axis=0)
        centered_data = arrow_pixels - mean_pixel

        _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
        direction_estimate_raw = Vt[0][::-1]
        direction_estimate = (direction_estimate_raw / np.linalg.norm(direction_estimate_raw)) * 10

        # Prevents direction from flipping 180 degrees
        print(f"Last direction {self.last_direction}\nCurrent direction {direction_estimate}")
        if np.linalg.norm(self.last_direction + direction_estimate) < 10:
            print("Switch!")
            direction_estimate = -direction_estimate

        pos_estimate = mean_pixel[::-1]

        return pos_estimate, direction_estimate
    
    def round(self, pos, direction):
        return np.round(pos).astype(int), np.round(direction).astype(int)
    
    def visualize(self, pos, direction):
        self.map_white = self.map_white_copy.copy()

        radius, color, thickness = 3, (0,255,0), -1

        cv.circle(self.map_white, pos, radius, color, thickness)
        cv.arrowedLine(self.map_white,pos, pos+direction,(255,0,0),3)
        self.map_white[self.nodes[:,0], self.nodes[:,1]] = np.array([0,0,255])

        print(f"Direction {direction}")
        print(f"Current position {pos}")

        return self.map_white
    
    def create_nodes(self):
        y_end, x_end, _ = self.map_white.shape
        x_coords = np.arange(0, x_end+1, 20)
        y_coords = np.arange(0, y_end+1, 20)
        x, y = np.meshgrid(x_coords, y_coords)
        coords = np.stack((x.flatten(), y.flatten()), axis=1)
        whites = np.argwhere(np.all(self.map_white==255, axis=-1))

        # Find intersection between coords and whites
        # More efficient methods can be looked up here:
        # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
        set1 = set(map(tuple, coords))
        set2 = set(map(tuple, whites))

        nodes = np.array(list(set1 & set2))
        return nodes
    
