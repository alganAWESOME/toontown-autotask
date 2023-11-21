import cv2 as cv
import numpy as np
from window_capture import WindowCapture
from time import sleep
from apply_filter import ApplyFilter
import pyautogui as pg

def main():
    global keys_pressed
    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    facing_x = wincap.w // 2

    facing_threshold = 30
    reached_threshold = 5000
    danger_threshold = 10000
    lost_threshold = 100
    searching = False
    danger = False

    pg.PAUSE = 0
    keys_pressed = {'up':True, 'left':False, 'right':False}
    walkable_det = ApplyFilter('ttc-street-walkable')
    streetlamp_det = ApplyFilter('ttc-streetlamp')

    sleep(2)
    pg.keyDown('up')
    while True:
        if wincap.screenshot is None:
            continue
        screenshot = wincap.screenshot

        streetlamps = streetlamp_det.apply(screenshot)
        walkable = walkable_det.apply(screenshot)
        target = find_largest_blob(streetlamps, 1)

        walkable_pixels = np.count_nonzero(walkable)
        #print(f"walkable_pixels={walkable_pixels}")

        target_coord = mean_coord(target)
        if target_coord is None and walkable_pixels < lost_threshold:
            searching = True
        else:
            searching = False

        if target_coord is not None:
            target_x, _ = target_coord
        else:
            try:
                target_x, _ = mean_coord(walkable_pixels)
            except:
                pass

        print(f"searching={searching}")

        if not searching:
            pg.keyUp('left')
            pg.keyDown('up')
            if target_x < facing_x - facing_threshold:
                pg.keyDown('left')
                pg.keyUp('right')
            elif target_x > facing_x + facing_threshold:
                pg.keyDown('right')
                pg.keyUp('left')
            else:
                pg.keyUp('left')
                pg.keyUp('right')
        else:
            pg.keyUp('up')
            pg.keyDown('left')
                
        cv.imshow("game",target)
        cv.imshow('walkable', walkable)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

def toggle_key(key, press):
    global keys_pressed

    if press:
        if not keys_pressed[key]:
            pg.keyDown(key)
            keys_pressed[key] = True
    else:
        if keys_pressed[key]:
            pg.keyUp(key)
            keys_pressed[key] = False

def mean_coord(image):
    # Ensure the image is in grayscale
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Get the indices of all pixels that are white (value 255)
    y_coords, x_coords = np.where(image == 255)

    # Calculate the mean coordinates
    if len(x_coords) > 0 and len(y_coords) > 0:
        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        return mean_x, mean_y
    else:
        return None  # Return None if there are no white pixels

def find_largest_blob(binary_image, n):
    # Find all contours in the binary image
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Calculate the area of each contour and sort them in descending order
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # Check if n is within the range of available contours
    if n - 1 >= len(contours) or n <= 0:
        return binary_image

    # Create a new blank image
    nth_largest_blob = np.zeros_like(binary_image)

    # Draw the n-th largest contour (n-1 in zero-indexed Python)
    cv.drawContours(nth_largest_blob, [contours[n - 1]], -1, (255, 255, 255), -1)

    return nth_largest_blob

if __name__ == "__main__":
    main()