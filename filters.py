import cv2 as cv
import numpy as np

class Trackbars:

    def __init__(self) -> None:
        self.window_name = "Trackbars"
        self.map = cv.imread('loopy-lane-map.png')

        cv.namedWindow(self.window_name)
        cv.createTrackbar("Threshold1", self.window_name, 0, 255, self.nothing)
        cv.createTrackbar("Threshold2", self.window_name, 0, 255, self.nothing)

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
        cv.createTrackbar('topleftX',self.window_name,0,640,self.nothing)
        cv.createTrackbar('topleftY',self.window_name,0,480,self.nothing)
        cv.createTrackbar('bottomrightX',self.window_name,0,640,self.nothing)
        cv.createTrackbar('bottomrightY',self.window_name,0,480,self.nothing)

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

        self.threshold1 = cv.getTrackbarPos('Threshold1',self.window_name)
        self.threshold2 = cv.getTrackbarPos('Threshold2',self.window_name)

        self.topleftX = cv.getTrackbarPos('topleftX',self.window_name)
        self.topleftY = cv.getTrackbarPos('topleftY',self.window_name)
        self.bottomrightX = cv.getTrackbarPos('bottomrightX',self.window_name)
        self.bottomrightY = cv.getTrackbarPos('bottomrightY',self.window_name)

        return self.apply_filter(screenshot)
    
    def apply_filter(self, screenshot):
        # To be optimized: filters should be applied to self.map beforehand
        # and saved as opposed to applied every frame.
        def crop_minimap(image):
            # Crops screenshot into the shape of self.map
            gray_image = cv.cvtColor(self.map, cv.COLOR_BGR2GRAY)
            _, mask = cv.threshold(gray_image, 1, 255, cv.THRESH_BINARY)
            mask_3channel = cv.merge((mask, mask, mask))
            cropped = cv.bitwise_and(image, mask_3channel)

            return cropped

        def hsv_filter(image):
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.lower, self.upper)
            hsv_filtered = cv.bitwise_and(image,image, mask= mask)
            return hsv_filtered

        def image_diff(image):
            # Finds image difference between screenshot and map
            kernel_size = (5, 5)
            blurred_image = cv.GaussianBlur(image, kernel_size, 0)
            blurred_map = cv.GaussianBlur(self.map, kernel_size, 0)
            diff_image = cv.absdiff(blurred_image, blurred_map)
            return cv.cvtColor(diff_image, cv.COLOR_BGR2GRAY)

        cropped = crop_minimap(screenshot)
        #hsv_filtered = hsv_filter(cropped)
        diff_image = image_diff(cropped)
        _, filtered = cv.threshold(diff_image, self.threshold1, 255, cv.THRESH_BINARY)

        #screenshot = screenshot[self.topleftY:self.bottomrightY, self.topleftX:self.bottomrightX]

        return filtered
    
    def read_filter_values(self):
        cv.setTrackbarPos('Threshold1',self.window_name,57)
        cv.setTrackbarPos('Threshold2',self.window_name,255)
        cv.setTrackbarPos('HMin',self.window_name,85)
        cv.setTrackbarPos('SMin',self.window_name,161)
        cv.setTrackbarPos('VMin',self.window_name,133)
        cv.setTrackbarPos('HMax',self.window_name,255)
        cv.setTrackbarPos('SMax',self.window_name,255)
        cv.setTrackbarPos('VMax',self.window_name,255)
        cv.setTrackbarPos('Canny1',self.window_name,255)
        cv.setTrackbarPos('Canny2',self.window_name,255)
        cv.setTrackbarPos('topleftX',self.window_name, 0)
        cv.setTrackbarPos('topleftY',self.window_name, 0)
        cv.setTrackbarPos('bottomrightX',self.window_name, 640)
        cv.setTrackbarPos('bottomrightY',self.window_name, 480)

    
    def nothing(self, x):
        pass