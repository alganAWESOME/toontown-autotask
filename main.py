import cv2 as cv
import numpy as np
from WindowCapture import WindowCapture
from Filters import Trackbars
from MinimapDetector import Detector
from Pathfinder import Pathfinder
from time import sleep

def main():
    trackbars = Trackbars()

    wincap = WindowCapture("Toontown Offline")
    wincap.start()
    
    arrow_detector = Detector()
    pathfinder = Pathfinder()

    sleep(2)
    while True:
        if wincap.screenshot is None:
            continue
        
        filtered = trackbars.read_trackbars_and_apply_filter(wincap.screenshot)
        pos, direction = arrow_detector.main(filtered)
        minimap = pathfinder.visualize(pos, direction)
        pathfinder.main(pos, direction)

        cv.imshow("game",minimap)
        #cv.imshow("filtered", filtered)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()