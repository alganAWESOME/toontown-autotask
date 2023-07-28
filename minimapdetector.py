import numpy as np

def detect_arrow_pos(filtered):
    # Returns (y,x) coords of arrow
    y_white, x_white = np.where(filtered==255)
    try:
        y_white, x_white = y_white[0], x_white[0]
    except:
        return

    y, x = np.mean(y_white), np.mean(x_white)

    return (int(x), int(y))
    