import cv2 as cv
import numpy as np

class Trackbars:

    def __init__(self) -> None:
        self.window_name = "Trackbars"

        cv.namedWindow(self.window_name)
        cv.createTrackbar("Threshold", self.window_name, 0, 255, self.nothing)

        # create trackbars for color change
        cv.createTrackbar('HMin',self.window_name,0,179,self.nothing) # Hue is from 0-179 for Opencv
        cv.createTrackbar('SMin',self.window_name,0,255,self.nothing)
        cv.createTrackbar('VMin',self.window_name,0,255,self.nothing)
        cv.createTrackbar('HMax',self.window_name,0,179,self.nothing)
        cv.createTrackbar('SMax',self.window_name,0,255,self.nothing)
        cv.createTrackbar('VMax',self.window_name,0,255,self.nothing)

        # Trackbars for canny edge detection
        cv.createTrackbar('Canny1',self.window_name,0,255,self.nothing)
        cv.createTrackbar('Canny2',self.window_name,0,255,self.nothing)

        # Trackbars for window position
        cv.createTrackbar('topleftX',self.window_name,10,100,self.nothing)
        cv.createTrackbar('topleftY',self.window_name,10,100,self.nothing)
        cv.createTrackbar('bottomrightX',self.window_name,10,100,self.nothing)
        cv.createTrackbar('bottomrightY',self.window_name,10,100,self.nothing)

        self.read_filter_values()
    
    def read_trackbars_and_apply_filter(self, screenshot):
        # get current positions of all trackbars
        hMin = cv.getTrackbarPos('HMin',self.window_name)
        sMin = cv.getTrackbarPos('SMin',self.window_name)
        vMin = cv.getTrackbarPos('VMin',self.window_name)
        hMax = cv.getTrackbarPos('HMax',self.window_name)
        sMax = cv.getTrackbarPos('SMax',self.window_name)
        vMax = cv.getTrackbarPos('VMax',self.window_name)

        # Set minimum and max HSV values to display
        self.lower = np.array([hMin, sMin, vMin])
        self.upper = np.array([hMax, sMax, vMax])

        self.canny1 = cv.getTrackbarPos('Canny1',self.window_name)
        self.canny2 = cv.getTrackbarPos('Canny2',self.window_name)

        self.threshold = cv.getTrackbarPos('Threshold',self.window_name)

        self.topleftX = cv.getTrackbarPos('topleftX',self.window_name)
        self.topleftY = cv.getTrackbarPos('topleftY',self.window_name)
        self.bottomrightX = cv.getTrackbarPos('bottomrightX',self.window_name)
        self.bottomrightY = cv.getTrackbarPos('bottomrightY',self.window_name)

        return self.apply_filter(screenshot)
    
    def apply_filter(self, screenshot):
        hsv = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower, self.upper)
        screenshot = cv.bitwise_and(screenshot,screenshot, mask= mask)

        screenshot = cv.Canny(screenshot, self.canny1, self.canny2)

        #_, screenshot = cv.threshold(screenshot, self.threshold, 255, cv.THRESH_BINARY)

        #screenshot = screenshot[self.topleftY:self.bottomrightY, self.topleftX:self.bottomrightX]

        return screenshot
    
    def read_filter_values(self):
        cv.setTrackbarPos('Threshold',self.window_name,0)
        cv.setTrackbarPos('HMin',self.window_name,85)
        cv.setTrackbarPos('SMin',self.window_name,161)
        cv.setTrackbarPos('VMin',self.window_name,133)
        cv.setTrackbarPos('HMax',self.window_name,255)
        cv.setTrackbarPos('SMax',self.window_name,255)
        cv.setTrackbarPos('VMax',self.window_name,255)
        cv.setTrackbarPos('Canny1',self.window_name,255)
        cv.setTrackbarPos('Canny2',self.window_name,255)

    
    def nothing(self, x):
        pass