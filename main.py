import cv2 as cv
import numpy as np
from window_capture import WindowCapture
from filters import Trackbars
from minimap_detector import Detector
from pathfinder import Pathfinder
from visualizer import Visualizer, GraphCreator
from time import sleep
from text_detector import TextDetector

def main():
    trackbars = Trackbars()

    wincap = WindowCapture("Toontown Offline")
    wincap.start()
    
    arrow_detector = Detector()
    pathfinder = Pathfinder(60)
    visualizer = Visualizer()
    graph_creator = GraphCreator(visualizer)
    ocr = TextDetector((4, 203, 137, 23))

    graph_creation_mode = True

    sleep(2)
    while True:
        if wincap.screenshot is None:
            continue

        filtered = trackbars.read_trackbars_and_apply_filter(wincap.screenshot)
        pos, direction = arrow_detector.main(filtered)
        minimap = visualizer.visualize(pos, direction)
        if graph_creation_mode:
            graph_creator.main(pos)
        else:
            pathfinder.main(pos, direction)
        
        processed_screenshot = ocr.detect_text(wincap.screenshot)
        cv.imshow("text", processed_screenshot)


        cv.imshow("game",minimap)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()