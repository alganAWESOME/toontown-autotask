import sys
sys.path.append(sys.path[0] + '/..')
import cv2 as cv
import numpy as np
from tkinter import *
from window_capture import WindowCapture
from filters import *

class GameColorFilter:
    def __init__(self, window_name):
        self.window_capture = WindowCapture(window_name)
        self.filters = []  # List of filter objects
        self.init_tkinter()
        self.current_screenshot = None

    def init_tkinter(self):
        self.root = Tk()
        self.root.title("Filter Configurator")

        # Button to add a new HSV filter
        Button(self.root, text="Add HSV Filter", command=self.add_hsv_filter).pack()

        # Listbox to show active filters
        self.filter_list = Listbox(self.root)
        self.filter_list.pack()

        # Button to configure the selected filter
        Button(self.root, text="Configure", command=self.configure_filter).pack()

    def add_hsv_filter(self):
        new_filter = HSVFilter()
        self.filters.append(new_filter)
        self.filter_list.insert(END, "HSV Filter")

    def configure_filter(self):
        selected_index = self.filter_list.curselection()
        if selected_index:
            selected_filter = self.filters[selected_index[0]]
            selected_filter.configure(self.root, self.update_filters)

    def update_filters(self):
        # Refreshes the filters; called as a callback from filter configuration
        self.apply_filters()

    def apply_filters(self):
        if self.current_screenshot is not None:
            filtered_image = self.current_screenshot.copy()
            for filter_obj in self.filters:
                filtered_image = filter_obj.apply(filtered_image)
            cv.imshow("Filtered", filtered_image)

    def start(self):
        self.window_capture.start()
        cv.namedWindow("Original")
        cv.setMouseCallback("Original", self.on_mouse_click)

        while True:
            self.current_screenshot = self.window_capture.screenshot
            if self.current_screenshot is None:
                continue

            cv.imshow("Original", self.current_screenshot)
            self.apply_filters()

            key = cv.waitKey(1)
            if key == ord('q'):
                self.window_capture.stop()
                cv.destroyAllWindows()
                break

            self.root.update()

    def on_mouse_click(self, event, x, y, flags, param):
        # Forward the mouse click event to the selected filter (if any)
        selected_index = self.filter_list.curselection()
        if selected_index and self.current_screenshot is not None:
            selected_filter = self.filters[selected_index[0]]
            if hasattr(selected_filter, 'on_mouse_click'):
                selected_filter.on_mouse_click(event, x, y, flags, param, self.current_screenshot)
                self.apply_filters()

def main():
    color_filter = GameColorFilter("Toontown Offline")
    color_filter.start()

if __name__ == "__main__":
    main()
