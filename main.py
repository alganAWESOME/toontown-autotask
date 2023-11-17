import cv2 as cv
from window_capture import WindowCapture
from time import sleep
from filters.apply_filters import ApplyFilter

def main():
    wincap = WindowCapture("Toontown Offline")
    wincap.start()
    filt = ApplyFilter('my_hsv')

    sleep(2)
    while True:
        if wincap.screenshot is None:
            continue
        
        filtered = filt.apply(wincap.screenshot)

        cv.imshow("game",wincap.screenshot)
        cv.imshow('filtered', filtered)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()