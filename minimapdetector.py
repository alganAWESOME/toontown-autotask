import numpy as np

class Detector:
    def __init__(self) -> None:
        self.last_direction = np.array([-10,0])
        #self.draw_nodes()

    def main(self, filtered):
        pos, direction = self.detect_arrow(filtered)
        self.last_direction = direction
        pos_rounded, direction_rounded = self.round(pos, direction)
        return pos_rounded, direction_rounded
    
    def detect_arrow(self, filtered):
        arrow_pixels = np.argwhere(filtered == 255)
        mean_pixel = np.mean(arrow_pixels, axis=0)
        centered_data = arrow_pixels - mean_pixel

        _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
        direction_estimate_raw = Vt[0][::-1]
        direction_estimate = (direction_estimate_raw / np.linalg.norm(direction_estimate_raw)) * 10

        # Prevents direction from flipping 180 degrees
        if np.linalg.norm(self.last_direction + direction_estimate) < 10:
            direction_estimate = -direction_estimate

        pos_estimate = mean_pixel[::-1]

        return pos_estimate, direction_estimate
    
    def round(self, pos, direction):
        return np.round(pos).astype(int), np.round(direction).astype(int)
    