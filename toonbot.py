import pyautogui as pg
from window_capture import WindowCapture
from states import *
import cv2 as cv

pg.PAUSE = 0

class ToonBot:
    def __init__(self):
        self.wincap = WindowCapture('Toontown Offline')
        self.wincap.start()
        self.facing_x = self.wincap.w // 2
        self.running = True
        self.kb_access = False

        self.state = StreetNavigation(self)
        
    def mainloop(self):
        while self.running:
            screenshot = self.wincap.screenshot
            if screenshot is None:
                continue

            self.state.update(screenshot)
            self.visualize()

        self.stop_moving()

    def visualize(self):
        for viz_name, viz in self.state.visualizations.items():
            try:
                cv.imshow(viz_name, viz)
            except Exception as e:
                print(f"visualisation_err={e}")
                
        key = cv.waitKey(1)
        if key == ord('q'):
            self.wincap.stop()
            cv.destroyAllWindows()
            self.running = False

    def start_moving(self):
        if self.kb_access:
            pg.keyDown('up')

    def stop_moving(self):
        if self.kb_access:
            pg.keyUp('up')

    def turn_right(self):
        if self.kb_access:
            pg.keyDown('right')
            pg.keyUp('left')

    def turn_left(self):
        if self.kb_access:
            pg.keyDown('left')
            pg.keyUp('right')

    def stop_turning(self):
        if self.kb_access:
            pg.keyUp('right')
            pg.keyUp('left')

    def toggle_minimap(self):
        if self.kb_access:
            pg.press('alt')

if __name__ == '__main__':
    bot = ToonBot()
    bot.mainloop()