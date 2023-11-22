import cv2 as cv
import numpy as np
from window_capture import WindowCapture
from time import sleep
from apply_filter import ApplyFilter
import pyautogui as pg
from random import randint

def main():
    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    facing_x = wincap.w // 2
    target_x = facing_x - 200

    cog_dangerous_thresh = 1800

    pg.PAUSE = 0

    streetlamp_det = ApplyFilter('mml-walkable-test')
    cog_detector = ApplyFilter('cogs')

    sleep(2)
    pg.keyDown('up')
    while True:
        if wincap.screenshot is None:
            continue
        screenshot = wincap.screenshot

        target = streetlamp_det.apply(screenshot)
        try:
            target_x, _ = mean_coord(target)
            searching = False
        except:
            searching = True

        cogs = cog_detector.apply(screenshot)
        contours, _ = cv.findContours(cogs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        # print([cv.contourArea(cont) for cont in contours])
        # print([np.abs(calc_centroid(cont)[0] - facing_x) for cont in contours])

        try:
            largest = contours[0]
            largest_area = cv.contourArea(largest)
            cog_x, cog_y = calc_centroid(largest)
        except:
            largest_area = 0
            cog_x = 0

        diff_x = cog_x - facing_x # if negative turn right
        if largest_area > 1800 and np.abs(diff_x) < 200:
            danger = True
        else:
            danger = False

        if searching:
            print("searching")
        if danger:
            print("danger")

        if True:
            if not danger:
                if not searching:
                    pg.keyDown('up')
                else:
                    pg.keyUp('up')

                if target_x < facing_x - 25:
                    pg.keyDown('left')
                    pg.keyUp('right')
                elif target_x > facing_x + 25:
                    pg.keyDown('right')
                    pg.keyUp('left')
                else:
                    pg.keyUp('right')
                    pg.keyUp('left')
            else:
                pg.keyDown('up')
                if diff_x < 0:
                    pg.keyDown('right')
                    pg.keyUp('left')
                else:
                    pg.keyDown('left')
                    pg.keyUp('right')
        else:
            pass
        viz = visualise([cogs, target])
        cv.imshow('visualisation', viz)
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
    
def calc_centroid(contour):
    M = cv.moments(contour)
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

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

def visualise(binary_images):
    colors = [(0,0,255), (255,0,0), (0, 255, 0), (0, 128, 255), (0,255,255), (0,255,128)]
    viz = np.zeros_like(cv.cvtColor(binary_images[0], cv.COLOR_GRAY2BGR))
    for i, img in enumerate(binary_images):
        viz[img != 0] = colors[i]
    return viz




if __name__ == "__main__":
    main()