import cv2 as cv
import numpy as np
from tkinter import *

class BaseFilter:
    def __init__(self):
        pass

    def configure(self):
        raise NotImplementedError

    def apply(self, image):
        raise NotImplementedError
    
class HSVFilter(BaseFilter):
    name = "HSV Filter"

    def __init__(self):
        super().__init__()
        self.hue_threshold = 20
        self.saturation_threshold = 20
        self.value_threshold = 20
        self.HSV_range = None
        self.clicked_color = None

    def configure(self, root, update_callback):
        configure_window = Toplevel(root)
        configure_window.title("Configure HSV Filter")
        
        # Each scale now triggers update_callback immediately after the value is changed
        Scale(configure_window, from_=0, to=89, orient=HORIZONTAL, label="Hue Threshold",
              command=lambda val: self.on_hue_threshold_change(val, update_callback)).pack()
        Scale(configure_window, from_=0, to=128, orient=HORIZONTAL, label="Saturation Threshold",
              command=lambda val: self.on_saturation_threshold_change(val, update_callback)).pack()
        Scale(configure_window, from_=0, to=128, orient=HORIZONTAL, label="Value Threshold",
              command=lambda val: self.on_value_threshold_change(val, update_callback)).pack()
        
    def on_mouse_click(self, event, x, y, flags, param, screenshot):
        if event == cv.EVENT_LBUTTONDOWN:
            # Extract the BGR color from the screenshot at the clicked position
            bgr_color = screenshot[y, x]
            self.clicked_color = bgr_color
            self.update_HSV_range()

    def update_HSV_range(self):
        if self.clicked_color is None:
            return

        # Convert the BGR color to HSV color space
        bgr_color = np.uint8([[self.clicked_color]])
        hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)[0][0]

        # Calculate the minimum and maximum HSV values
        hue_min = max(hsv_color[0] - self.hue_threshold, 0)
        hue_max = min(hsv_color[0] + self.hue_threshold, 180)
        sat_min = max(hsv_color[1] - self.saturation_threshold, 0)
        sat_max = min(hsv_color[1] + self.saturation_threshold, 255)
        val_min = max(hsv_color[2] - self.value_threshold, 0)
        val_max = min(hsv_color[2] + self.value_threshold, 255)

        self.HSV_range = [(hue_min, sat_min, val_min), (hue_max, sat_max, val_max)]

    def apply(self, image):
        if self.HSV_range is None:
            return image

        lower_bound, upper_bound = self.HSV_range
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))
        return cv.bitwise_and(image, image, mask=mask)

    def on_hue_threshold_change(self, val, update_callback):
        self.hue_threshold = int(val)
        self.update_HSV_range()
        update_callback()

    def on_saturation_threshold_change(self, val, update_callback):
        self.saturation_threshold = int(val)
        self.update_HSV_range()
        update_callback()

    def on_value_threshold_change(self, val, update_callback):
        self.value_threshold = int(val)
        self.update_HSV_range()
        update_callback()

class ContrastFilter(BaseFilter):
    name = "Contrast Filter"

    def __init__(self):
        super().__init__()
        self.contrast_level = 1.0  # Default contrast level

    def configure(self, root, update_callback):
        configure_window = Toplevel(root)
        configure_window.title("Configure Contrast Filter")

        Scale(configure_window, from_=0, to=3.0, resolution=0.1, orient=HORIZONTAL, label="Contrast Level",
              command=lambda val: self.on_contrast_change(val, update_callback)).pack()

    def on_contrast_change(self, val, update_callback):
        self.contrast_level = float(val)
        update_callback()

    def apply(self, image):
        # Convert to float for more precision for transformations
        image_float = image.astype(np.float32)

        # Apply the contrast formula
        image_adjusted = image_float * self.contrast_level

        # Clip values to the valid range (0 to 255) and convert back to uint8
        image_adjusted = np.clip(image_adjusted, 0, 255).astype(np.uint8)

        return image_adjusted
