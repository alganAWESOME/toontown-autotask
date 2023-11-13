import cv2 as cv
from window_capture import WindowCapture
from time import sleep

def main():
    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    sleep(2)
    while True:
        if wincap.screenshot is None:
            continue

        cv.imshow("game",wincap.screenshot)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()