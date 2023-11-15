import sys
sys.path.append(sys.path[0] + '/..')
import cv2 as cv
import numpy as np
from tkinter import *
from window_capture import WindowCapture

class GameColorFilter:
    def __init__(self, window_name):
        self.window_capture = WindowCapture(window_name)
        self.filter_options = ['HSV Filter', 'Contrast Filter']
        self.filters = []  # List of filter objects
        self.init_tkinter()
        self.current_screenshot = None

    def init_tkinter(self):
        self.root = Tk()
        self.root.title("Filter Configurator")

        # Button to add a new HSV filter
        Button(self.root, text="Add Filter", command=self.add_filter).pack()

        # Listbox to show active filters
        self.filter_list = Listbox(self.root)
        self.filter_list.pack()

        # Button to configure the selected filter
        Button(self.root, text="Configure", command=self.configure_filter).pack()

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
        from filters import HSVFilter, ContrastFilter
        print(f"filter_name={filter_name}")
        if filter_name == "HSV Filter":
            new_filter = HSVFilter()
        elif filter_name == "Contrast Filter":
            new_filter = ContrastFilter()
        else:
            raise ValueError("Unknown filter type")

        self.filters.append(new_filter)
        self.filter_list.insert(END, filter_name)
        filter_window.destroy()

    def configure_filter(self):
        selected_index = self.filter_list.curselection()
        if selected_index:
            selected_filter = self.filters[selected_index[0]]
            selected_filter.configure(self.root, self.update_filters)

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
