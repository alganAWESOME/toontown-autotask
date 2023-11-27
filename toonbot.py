import pyautogui as pg
from window_capture import WindowCapture
from states import *

pg.PAUSE = 0

class ToonBot:
    def __init__(self) -> None:
        self.wincap = WindowCapture('Toontown Offline')
        self.wincap.start()
        self.facing_x = self.wincap.w // 2
        self.state = StreetNavigation(self)

    def mainloop(self):
        while True:
            screenshot = self.wincap.screenshot
            if screenshot is None:
                continue

            self.state.update(screenshot)

    def start_moving(self):
        pg.keyDown('up')

    def stop_moving(self):
        pg.keyUp('up')

    def turn_right(self):
        pg.keyDown('right')
        pg.keyUp('left')

    def turn_left(self):
        pg.keyDown('left')
        pg.keyUp('right')

    def stop_turning(self):
        pg.keyUp('right')
        pg.keyUp('left')

    def toggle_minimap(self):
        pg.press('alt')

if __name__ == '__main__':
    bot = ToonBot()
    bot.mainloop()