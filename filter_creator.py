import cv2 as cv
import numpy as np
import json
from tkinter import *
from window_capture import WindowCapture

class GameColorFilter:
    def __init__(self, window_name):
        self.window_capture = WindowCapture(window_name)
        self.clicked_color = None
        self.threshold = 20
        self.HSV_range = None
        self.init_tkinter()

    def init_tkinter(self):
        self.root = Tk()
        self.root.title("Filter Configurator")
        Label(self.root, text="Filter Name:").pack()
        self.filter_name_entry = Entry(self.root)
        self.filter_name_entry.pack()
        Button(self.root, text="Append to JSON", command=self.append_to_json).pack()
        Button(self.root, text="Rewrite JSON", command=self.rewrite_json).pack()

    def start(self):
        self.window_capture.start()
        cv.namedWindow("Original")
        cv.createTrackbar("Threshold", "Original", self.threshold, 100, self.on_threshold_change)
        cv.setMouseCallback("Original", self.on_mouse_click)

        while True:
            screenshot = self.window_capture.screenshot
            if screenshot is None:
                continue

            cv.imshow("Original", screenshot)
            if self.HSV_range is not None:
                filtered_image = self.apply_color_filter(screenshot)
                cv.imshow("Filtered", filtered_image)

            key = cv.waitKey(1)
            if key == ord('q'):
                self.window_capture.stop()
                cv.destroyAllWindows()
                break

            self.root.update()

    def on_threshold_change(self, pos):
        self.threshold = pos
        if self.clicked_color is not None:
            self.update_HSV_range()

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.clicked_color = self.window_capture.screenshot[y, x]
            self.update_HSV_range()

    def update_HSV_range(self):
        hsv_color = cv.cvtColor(np.uint8([[self.clicked_color]]), cv.COLOR_BGR2HSV)[0][0]
        self.HSV_range = {
            "HSV_min": [max(hsv_color[0] - self.threshold, 0), max(hsv_color[1] - self.threshold, 0), max(hsv_color[2] - self.threshold, 0)],
            "HSV_max": [min(hsv_color[0] + self.threshold, 179), min(hsv_color[1] + self.threshold, 255), min(hsv_color[2] + self.threshold, 255)]
        }

    def apply_color_filter(self, image):
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_bound = np.array(self.HSV_range["HSV_min"])
        upper_bound = np.array(self.HSV_range["HSV_max"])
        mask = cv.inRange(hsv_image, lower_bound, upper_bound)
        return cv.bitwise_and(image, image, mask=mask)

    def append_to_json(self):
        self.update_json(append=True)

    def rewrite_json(self):
        self.update_json(append=False)

    def update_json(self, append=True):
        filter_name = self.filter_name_entry.get()
        if self.HSV_range and filter_name:
            # Convert NumPy int32 values to Python int
            hsv_min = [int(val) for val in self.HSV_range["HSV_min"]]
            hsv_max = [int(val) for val in self.HSV_range["HSV_max"]]

            new_filter = {
                "name": filter_name,
                "color_ranges": [
                    {
                        "HSV_min": hsv_min,
                        "HSV_max": hsv_max
                    }
                ]
            }

            with open('filters.json', 'r+' if append else 'w') as file:
                data = {"filters": []}
                if append:
                    data = json.load(file)

                if filter_name in [filt['name'] for filt in data['filters']]:
                    for filt in data['filters']:
                        if filt['name'] == filter_name:
                            filt['color_ranges'].append(new_filter["color_ranges"][0])
                else:
                    data["filters"].append(new_filter)

                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

def main():
    color_filter = GameColorFilter("Toontown Offline")
    color_filter.start()

if __name__ == "__main__":
    main()
