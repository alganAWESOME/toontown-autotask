import cv2 as cv
import numpy as np
from windowcapture import WindowCapture

def main():
    TRACKBAR_WINDOW = "Trackbars"
    cv.namedWindow(TRACKBAR_WINDOW)
    cv.createTrackbar("Threshold", TRACKBAR_WINDOW, 0, 255, nothing)
    # create trackbars for color change
    cv.createTrackbar('HMin',TRACKBAR_WINDOW,0,179,nothing) # Hue is from 0-179 for Opencv
    cv.createTrackbar('SMin',TRACKBAR_WINDOW,0,255,nothing)
    cv.createTrackbar('VMin',TRACKBAR_WINDOW,0,255,nothing)
    cv.createTrackbar('HMax',TRACKBAR_WINDOW,0,179,nothing)
    cv.createTrackbar('SMax',TRACKBAR_WINDOW,0,255,nothing)
    cv.createTrackbar('VMax',TRACKBAR_WINDOW,0,255,nothing)

    # Set default value for MAX HSV trackbars.
    cv.setTrackbarPos('HMax', TRACKBAR_WINDOW, 179)
    cv.setTrackbarPos('SMax', TRACKBAR_WINDOW, 255)
    cv.setTrackbarPos('VMax', TRACKBAR_WINDOW, 255)

    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    while True:
        if wincap.screenshot is None:
            continue

        threshold = cv.getTrackbarPos("Threshold", TRACKBAR_WINDOW)
        # get current positions of all trackbars
        hMin = cv.getTrackbarPos('HMin',TRACKBAR_WINDOW)
        sMin = cv.getTrackbarPos('SMin',TRACKBAR_WINDOW)
        vMin = cv.getTrackbarPos('VMin',TRACKBAR_WINDOW)

        hMax = cv.getTrackbarPos('HMax',TRACKBAR_WINDOW)
        sMax = cv.getTrackbarPos('SMax',TRACKBAR_WINDOW)
        vMax = cv.getTrackbarPos('VMax',TRACKBAR_WINDOW)

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        window = apply_filter(wincap.screenshot, threshold, lower, upper)
        
        cv.imshow("screen", window)
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