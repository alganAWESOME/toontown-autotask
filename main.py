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

        screenshot = trackbars.read_trackbar_and_apply_filter(wincap.screenshot)
        
        cv.imshow("screen", screenshot)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

def apply_filter(screenshot, threshold, lower, upper):
    #screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    #_, screenshot = cv.threshold(screenshot, threshold, 255, cv.THRESH_BINARY)

    # Create HSV Image and threshold into a range.
    hsv = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    screenshot = cv.bitwise_and(screenshot,screenshot, mask= mask)

    return screenshot

def nothing(x):
    pass

main()