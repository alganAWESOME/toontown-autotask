import cv2 as cv
import numpy as np
from windowcapture import WindowCapture

wincap = WindowCapture("Toontown Offline")
wincap.start()

def apply_filter(screenshot):
    return screenshot

while True:
    if wincap.screenshot is None:
        continue
    
    window = apply_filter(wincap.screenshot)
    
    cv.imshow("screen", window)
    key = cv.waitKey(1)
    if key == ord('q'):
        wincap.stop()
        cv.destroyAllWindows()
        break

