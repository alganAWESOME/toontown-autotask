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
        pos = self.detect_arrow_pos(filtered)
        self.save_previous_position(pos)
        direction = self.detect_arrow_direction()
        pos_rounded, direction_rounded = self.round(pos, direction)
        minimap = self.draw_circle_on_map(pos_rounded, direction_rounded)
        return minimap
    
    def detect_arrow_pos(self,filtered):
        # Takes in filtered screenshot, returns (x,y) coords of arrow
        y_white, x_white = np.where(filtered==255)
        try:
            y_white, x_white = y_white[0], x_white[0]
        except:
            return

        coords = np.array([np.mean(x_white), np.mean(y_white)])

        return coords
    
    def save_previous_position(self, pos):
        # np.vstack doesn't work if self.last_frames is empty
        if len(self.last_frames) == 0:
            self.last_frames = np.array([pos])
        else:
            self.last_frames = np.vstack((self.last_frames.copy(), pos))
            if len(self.last_frames) == self.frame_count+1:
                self.last_frames = self.last_frames[1:].copy()
    
    def detect_arrow_direction(self):
        # Mean of previous frames
        # self.frames == [old_old, old, current]
        print(f"self.last_frames: {self.last_frames}")

        if len(self.last_frames) >= 2:
            pos_new = self.last_frames[-1]
            frame_count = int(len(self.last_frames)/2)
            pos_old = np.mean(self.last_frames[:-frame_count], axis=0)
            direction_raw = pos_new - pos_old
            if np.all(direction_raw == 0):
                direction = np.array([0,0])
            else:
                direction = (direction_raw / np.linalg.norm(direction_raw)) * 10
        else:
            direction = np.array([0,0])

        return direction
    
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