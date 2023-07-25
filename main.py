import cv2 as cv
import numpy as np
from windowcapture import WindowCapture
from filters import Trackbars

def main():
    trackbars = Trackbars()

    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    initial = wincap.screenshot

    while True:
        if wincap.screenshot is None:
            continue

        #screenshot = trackbars.read_trackbars_and_apply_filter(wincap.screenshot)
        
        # Compute the absolute difference between the two images
        diff_image = cv.absdiff(wincap.screenshot, initial)
        print(wincap.screenshot.shape)

        # Convert the difference image to grayscale
        screenshot = cv.cvtColor(diff_image, cv.COLOR_BGR2GRAY)
        
        cv.imshow("screen", screenshot)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

main()