import cv2 as cv
import numpy as np
from window_capture import WindowCapture
from time import sleep
from apply_filter import ApplyFilter
import pyautogui as pg

def main():
    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    facing_x = wincap.w // 2
    turn_threshold = 30
    reached_threshold = 20000
    searching_target = False

    pg.PAUSE = 0
    up_pressed, left_pressed, right_pressed = True, False, False

    walkable_det = ApplyFilter('ttc-street-walkable')

    sleep(2)
    pg.keyDown('up')
    while True:
        if wincap.screenshot is None:
            continue
        screenshot = wincap.screenshot

        target = walkable_det.apply(screenshot)

        coord = mean_coord(target)
        if coord is None:
            searching_target = True
        else:
            target_x, _ = coord
            searching_target = False

        if not searching_target:
            pg.keyDown('up')
            if facing_x < target_x - turn_threshold:
                pg.keyUp('left')
                pg.keyDown('right')
            elif facing_x > target_x + turn_threshold:    
                pg.keyUp('right')
                pg.keyDown('left')
            else:
                pg.keyUp('right')
                pg.keyUp('left')
            print(f'off_center_by={facing_x - target_x}')
        else:    
            pg.keyUp('up')
            pg.keyUp('right')
            pg.keyDown('left')
                
        cv.imshow("game",target)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

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

if __name__ == "__main__":
    main()