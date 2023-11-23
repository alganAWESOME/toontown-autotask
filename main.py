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
    arrow_detector = ApplyFilter('punchlineplace-arrow')

    direction = np.array([1,0])

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

        # if searching:
        #     print("searching")
        # if danger:
        #     print("danger")

        if False:
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
        #viz = visualise([cogs, target])
        #cv.imshow('visualisation', viz)

        arrow_viz = np.zeros_like(screenshot)
        arrow = arrow_detector.apply(screenshot)

        pos, direction = detect_arrow(arrow, direction)
        color=(255, 0, 255)
        thickness=2
        endpoint = (1*pos[0] + 1*direction[0], 1*pos[1] + 1*direction[1])
        cv.arrowedLine(arrow_viz, pos, endpoint, color, thickness)
        cv.imshow('arrow', arrow_viz)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

def detect_arrow(filtered, prev_direction):
    arrow_pixels = np.argwhere(filtered == 255)
    mean_pixel = np.mean(arrow_pixels, axis=0)
    centered_data = arrow_pixels - mean_pixel

    _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
    direction_estimate_raw = Vt[0][::-1]
    direction_estimate = (direction_estimate_raw / np.linalg.norm(direction_estimate_raw)) * 10

    # Prevents direction from flipping 180 degrees
    if np.linalg.norm(prev_direction + direction_estimate) < 10:
        direction_estimate = -direction_estimate

    pos_estimate = mean_pixel[::-1]

    return tuple(map(int, pos_estimate)), tuple(map(int, direction_estimate))

def detect_arrow_new(filtered):
    arrow_pixels = np.argwhere(filtered == 255)
    mean_pixel = np.mean(arrow_pixels, axis=0)
    centered_data = arrow_pixels - mean_pixel

    _, _, Vt = np.linalg.svd(centered_data, full_matrices=False)
    principal_component = Vt[0]

    # Calculate the projections on the principal component
    projections = np.dot(centered_data, principal_component)
    print(f"len_projects={len(projections)}")

    # Unweighted mean of projections
    unweighted_mean = np.mean(projections)

    # Weighted mean of projections
    unique, counts = np.unique(projections, return_counts=True)
    weighted_mean = np.sum(unique * counts) / np.sum(counts)

    # Map the means back to original space
    unweighted_mean_coord = mean_pixel + unweighted_mean * principal_component
    weighted_mean_coord = mean_pixel + weighted_mean * principal_component

    # Calculate the directional vector
    direction_vector = weighted_mean_coord - unweighted_mean_coord
    normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)

    pos_estimate = mean_pixel[::-1]

    print(f"pos={pos_estimate}")
    print(f"direction={normalized_direction_vector}")

    return tuple(map(int, pos_estimate)), tuple(map(int, normalized_direction_vector))

def detect_arrow_pos_and_direction(arrow_image):
    # Find coordinates of all white pixels
    y_coords, x_coords = np.where(arrow_image == 255)

    # Calculate the unweighted mean position
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    unweighted_mean = np.array([mean_x, mean_y])

    # Center the coordinates
    x_coords_centered = x_coords - mean_x
    y_coords_centered = y_coords - mean_y

    # Compute covariance matrix
    coords = np.vstack([x_coords_centered, y_coords_centered])
    covariance_matrix = np.cov(coords)

    # Compute principal component
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    principal_component_normalized = principal_component / np.linalg.norm(principal_component)

    # Project white pixels onto the principal axis
    projections = np.dot(coords.T, principal_component_normalized)

    # Calculate the weighted mean along the principal axis
    weighted_mean_projection = np.mean(projections)
    weighted_mean = unweighted_mean + weighted_mean_projection * principal_component_normalized

    # Calculate direction from unweighted mean to weighted mean
    arrow_direction = weighted_mean - unweighted_mean
    arrow_direction_normalized = arrow_direction / np.linalg.norm(arrow_direction)

    return (int(mean_x), int(mean_y)), tuple(map(int, principal_component_normalized))

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