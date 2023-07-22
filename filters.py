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
    
        # Set default value for MAX HSV trackbars.
        cv.setTrackbarPos('HMax', self.window_name, 179)
        cv.setTrackbarPos('SMax', self.window_name, 255)
        cv.setTrackbarPos('VMax', self.window_name, 255)
    
    def read_trackbar_and_apply_filter(self, screenshot):
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

        return self.apply_filter(screenshot)
    
    def apply_filter(self, screenshot):
        hsv = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower, self.upper)
        screenshot = cv.bitwise_and(screenshot,screenshot, mask= mask)

        return screenshot
    
    def nothing(self, x):
        pass