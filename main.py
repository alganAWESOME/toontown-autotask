import cv2 as cv
import numpy as np
from windowcapture import WindowCapture
from filters import Trackbars
from minimapdetector import Detector

def main():
    trackbars = Trackbars()

    wincap = WindowCapture("Toontown Offline")
    wincap.start()
    
    det = Detector()

    while True:
        if wincap.screenshot is None:
            continue
        
        filtered = trackbars.read_trackbars_and_apply_filter(wincap.screenshot)

        minimap = det.main(filtered)

        cv.imshow("game",minimap)
        
        cv.imshow("filtered", filtered)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

main()