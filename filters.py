import cv2
import numpy as np
import json

class Filter:
    def __init__(self, json_path='filters.json'):
        # Load filter parameters from the JSON file
        with open(json_path, 'r') as file:
            self.filters = json.load(file)["filters"]

    def apply_filter(self, image, filter_name):
        # Find the filter in the loaded JSON data
        for filter in self.filters:
            if filter["name"] == filter_name:
                # Apply all color ranges for the filter
                mask = None
                for color_range in filter["color_ranges"]:
                    lower = np.array(color_range["HSV_min"], dtype="uint8")
                    upper = np.array(color_range["HSV_max"], dtype="uint8")
                    temp_mask = cv2.inRange(image, lower, upper)
                    mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)
                return cv2.bitwise_and(image, image, mask=mask)
        return None  # Return None if the filter is not found

    def main(self, screenshot):
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Apply a filter (e.g., 'ttc_walkable') to the screenshot
        filtered_image = self.apply_filter(screenshot, 'ttc_walkable')

        return filtered_image