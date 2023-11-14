import cv2 as cv
import numpy as np
import json
from tkinter import *
from window_capture import WindowCapture

class GameColorFilter:
    def __init__(self, window_name):
        self.window_capture = WindowCapture(window_name)
        self.clicked_color = None
        self.hue_threshold = 20
        self.saturation_threshold = 20
        self.value_threshold = 20
        self.HSV_range = None
        self.init_tkinter()

    def init_tkinter(self):
        self.root = Tk()
        self.root.title("Filter Configurator")
        
        # Sliders for HSV threshold
        self.hue_scale = Scale(self.root, from_=0, to=89, orient=HORIZONTAL, label="Hue Threshold", command=self.on_hue_threshold_change)
        self.hue_scale.set(self.hue_threshold) 
        self.hue_scale.pack()
        
        self.saturation_scale = Scale(self.root, from_=0, to=255, orient=HORIZONTAL, label="Saturation Threshold", command=self.on_saturation_threshold_change)
        self.saturation_scale.set(self.saturation_threshold) 
        self.saturation_scale.pack()
        
        self.value_scale = Scale(self.root, from_=0, to=255, orient=HORIZONTAL, label="Value Threshold", command=self.on_value_threshold_change)
        self.value_scale.set(self.value_threshold) 
        self.value_scale.pack()

        Label(self.root, text="Filter Name:").pack()
        self.filter_name_entry = Entry(self.root)
        self.filter_name_entry.pack()
        Button(self.root, text="Append to JSON", command=self.append_to_json).pack()
        Button(self.root, text="Rewrite JSON", command=self.rewrite_json).pack()

    def start(self):
        self.window_capture.start()
        cv.namedWindow("Original")
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

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Capture the BGR color from the screenshot at the clicked position
            bgr_color = self.window_capture.screenshot[y, x]

            self.clicked_color = bgr_color

            # Update the HSV range based on the new color
            self.update_HSV_range()

    def update_HSV_range(self):
        if self.clicked_color is not None:
            # Convert the BGR color to HSV color space
            bgr_color = np.uint8([[self.clicked_color]])
            hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)[0][0]
            #print(f"hsv={hsv_color}")

            # Calculate the minimum and maximum HSV values, wrapping the Hue
            hue_min = (hsv_color[0] - self.hue_threshold) % 180
            hue_max = (hsv_color[0] + self.hue_threshold) % 180
            sat_min = max(hsv_color[1] - self.saturation_threshold, 0)
            sat_max = min(hsv_color[1] + self.saturation_threshold, 255)
            val_min = max(hsv_color[2] - self.value_threshold, 0)
            val_max = min(hsv_color[2] + self.value_threshold, 255)

            # If the range includes the wrap-around, split into two ranges
            if hue_min > hue_max:
                self.HSV_range = [
                    {"HSV_min": [hue_min, sat_min, val_min], "HSV_max": [179, sat_max, val_max]},
                    {"HSV_min": [0, sat_min, val_min], "HSV_max": [hue_max, sat_max, val_max]}
                ]
            else:
                self.HSV_range = [{"HSV_min": [hue_min, sat_min, val_min], "HSV_max": [hue_max, sat_max, val_max]}]


    def apply_color_filter(self, image):
        # Initialize a mask for all zeros (no pixels selected)
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Convert the image to HSV color space one time for efficiency
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Apply each filter in the HSV range list
        for filter_range in self.HSV_range:
            # Create mask for current filter range
            lower_bound = np.array(filter_range["HSV_min"])
            upper_bound = np.array(filter_range["HSV_max"])
            current_mask = cv.inRange(hsv_image, lower_bound, upper_bound)
            
            # Combine the current mask with the final mask using bitwise OR
            final_mask = cv.bitwise_or(final_mask, current_mask)

        # Apply the final mask to the original image
        filtered_image = cv.bitwise_and(image, image, mask=final_mask)
        return filtered_image


    def on_hue_threshold_change(self, val):
        self.hue_threshold = int(val)
        self.update_HSV_range()

    def on_saturation_threshold_change(self, val):
        self.saturation_threshold = int(val)
        self.update_HSV_range()

    def on_value_threshold_change(self, val):
        self.value_threshold = int(val)
        self.update_HSV_range()

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
