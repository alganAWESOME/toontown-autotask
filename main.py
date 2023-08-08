import cv2 as cv
import numpy as np
from window_capture import WindowCapture
from filters import Trackbars
from minimap_detector import Detector
from pathfinder import Pathfinder
from visualizer import Visualizer
from time import sleep

def main():
    trackbars = Trackbars()

    wincap = WindowCapture("Toontown Offline")
    wincap.start()
    
    arrow_detector = Detector()
    pathfinder = Pathfinder(40)
    visualizer = Visualizer()

    sleep(2)
    while True:
        if wincap.screenshot is None:
            continue
        
        filtered = trackbars.read_trackbars_and_apply_filter(wincap.screenshot)
        pos, direction = arrow_detector.main(filtered)
        #visualizer.graph_creator(pos)
        minimap = visualizer.visualize(pos, direction)
        pathfinder.main(pos, direction)

        cv.imshow("game",minimap)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()