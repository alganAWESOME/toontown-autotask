import cv2 as cv
import numpy as np
from windowcapture import WindowCapture
from filters import Trackbars
import minimapdetector as det

def main():
    trackbars = Trackbars()

    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    while True:
        if wincap.screenshot is None:
            continue

        filtered = trackbars.read_trackbars_and_apply_filter(wincap.screenshot)

        screenshot_copy = wincap.screenshot.copy()
        pos = det.detect_arrow_pos(filtered)
        # Draw a circle at the specified position
        radius = 3
        color = (0, 255, 0)  # Green color (in BGR format)
        thickness = -1  # Fill the circle (-1), set to a positive value for a circle outline
        cv.circle(screenshot_copy, pos, radius, color, thickness)

        cv.imshow("game",screenshot_copy)
        
        cv.imshow("filtered", filtered)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

main()