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
        self.filter_options = ['HSV Filter', 'Contrast Filter', 'Saturation']
        self.filters = []  # List of filter objects
        self.init_tkinter()
        self.current_screenshot = None
        self.current_filtered_image = None

    def init_tkinter(self):
        self.root = Tk()
        self.root.title("Filter Configurator")

        # Button to add a new HSV filter
        Button(self.root, text="Add Filter", command=self.add_filter).pack()

        # Listbox to show active filters
        self.filter_list = Listbox(self.root)
        self.filter_list.pack()

        # Frame for filter configuration
        self.config_frame = Frame(self.root)
        self.config_frame.pack(side=BOTTOM)
        self.filter_list.bind('<<ListboxSelect>>', lambda e: self.update_config_frame())

        # Add "Move Up" and "Move Down" buttons
        Button(self.root, text="Move Up", command=self.move_filter_up).pack()
        Button(self.root, text="Move Down", command=self.move_filter_down).pack()

        Button(self.root, text='Delete', command=self.delete_filter).pack()

    def add_filter(self):
        filter_window = Toplevel(self.root)
        filter_window.title("Select Filter Type")

        # Dynamically create buttons for each filter type
        for option in self.filter_options: 
            Button(filter_window, text=option, 
                   command=lambda name=option: self.create_filter(name, filter_window)).pack()

    def create_filter(self, filter_name, filter_window):
        if filter_name == "HSV Filter":
            new_filter = HSVFilter()
        elif filter_name == "Contrast Filter":
            new_filter = ContrastFilter()
        elif filter_name == 'Saturation':
            new_filter = SaturationFilter()
        else:
            raise ValueError("Unknown filter type")

        self.filters.append(new_filter)
        self.filter_list.insert(END, filter_name)
        filter_window.destroy()

    def update_config_frame(self):
        # Clear current configuration frame
        for widget in self.config_frame.winfo_children():
            widget.destroy()

        selected_index = self.filter_list.curselection()
        if selected_index:
            selected_filter = self.filters[selected_index[0]]
            selected_filter.configure(self.config_frame, self.update_filters)

    def move_filter_up(self):
        selected_index = self.filter_list.curselection()
        if selected_index and selected_index[0] > 0:
            index = selected_index[0]
            # Swap the filters
            self.filters[index], self.filters[index - 1] = self.filters[index - 1], self.filters[index]
            # Update the listbox
            self.filter_list.insert(index - 1, self.filter_list.get(index))
            self.filter_list.delete(index + 1)
            self.filter_list.select_set(index - 1)

    def move_filter_down(self):
        selected_index = self.filter_list.curselection()
        if selected_index and selected_index[0] < len(self.filters) - 1:
            index = selected_index[0]
            # Swap the filters
            self.filters[index], self.filters[index + 1] = self.filters[index + 1], self.filters[index]
            # Update the listbox
            self.filter_list.insert(index + 2, self.filter_list.get(index))
            self.filter_list.delete(index)
            self.filter_list.select_set(index + 1)

    def delete_filter(self):
        selected_index = self.filter_list.curselection()
        if selected_index:
            self.filters.pop(selected_index[0])
            self.filter_list.delete(selected_index[0])

    def update_filters(self):
        # Refreshes the filters; called as a callback from filter configuration
        self.apply_filters()

    def apply_filters(self):
        if self.current_screenshot is not None:
            self.current_filtered_image = self.current_screenshot.copy()
            for filter_obj in self.filters:
                self.current_filtered_image = filter_obj.apply(self.current_filtered_image)
            cv.imshow("Filtered", self.current_filtered_image)

    def start(self):
        self.window_capture.start()
        cv.namedWindow("Original")
        cv.namedWindow("Filtered")
        cv.setMouseCallback("Original", self.on_mouse_click_original)
        cv.setMouseCallback("Filtered", self.on_mouse_click_filtered)

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

    def on_mouse_click_original(self, event, x, y, flags, param):
        self.on_mouse_click(event, x, y, flags, param, source="original")

    def on_mouse_click_filtered(self, event, x, y, flags, param):
        self.on_mouse_click(event, x, y, flags, param, source="filtered")
    
    def on_mouse_click(self, event, x, y, flags, param, source):
        # Handle the mouse click event based on the source (original or filtered)
        selected_index = self.filter_list.curselection()
        if selected_index and (self.current_screenshot is not None or self.current_filtered_image is not None):
            selected_filter = self.filters[selected_index[0]]
            if hasattr(selected_filter, 'on_mouse_click'):
                # Pass the appropriate image based on the source
                source_image = self.current_filtered_image if source == "filtered" else self.current_screenshot
                selected_filter.on_mouse_click(event, x, y, flags, param, source_image)
                self.apply_filters()

def main():
    color_filter = GameColorFilter("Toontown Offline")
    color_filter.start()

if __name__ == "__main__":
    main()
