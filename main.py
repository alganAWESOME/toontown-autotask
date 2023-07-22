import cv2 as cv
import numpy as np
from windowcapture import WindowCapture
from filters import Trackbars

def main():
    trackbars = Trackbars()

    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    while True:
        if wincap.screenshot is None:
            continue

        screenshot = trackbars.read_trackbars_and_apply_filter(wincap.screenshot)
        
        cv.imshow("screen", screenshot)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

main()