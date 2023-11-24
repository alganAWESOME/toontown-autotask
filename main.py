import cv2 as cv
import numpy as np
from window_capture import WindowCapture
from time import sleep
from apply_filter import ApplyFilter
import pyautogui as pg
import json
import networkx as nx
from math import sqrt
import time

def main():
    wincap = WindowCapture("Toontown Offline")
    wincap.start()

    facing_x = wincap.w // 2
    target_x = facing_x - 200

    cog_dangerous_thresh = 1800

    pg.PAUSE = 0

    walkable_detector = ApplyFilter('ttc-street-walkable')
    cog_detector = ApplyFilter('cogs')
    arrow_detector = ApplyFilter('punchlineplace-arrow')

    danger = False

    # minimap related
    destination_node = 0
    looking_at_minimap = True
    directions = {'right':np.array([1,0]), 'left':np.array([-1,0]), 'up':np.array([0,-1]), 'down':np.array([0,1])}
    pos, direction = None, directions['right']
    minimap_graph = load_graph_to_networkx('graphs.json', "ttc-punchlineplace")
    path_calculated = False
    path_viz = None
    angle_thresh = (1/12) * np.pi # if angle below (-this/2) turn left
    node_reached_thresh = 10
    target_node = None

    sleep(2)
    pg.keyDown('up')
    pg.press('alt')
    last_time = time.time()
    while True:
        if wincap.screenshot is None:
            continue
        screenshot = wincap.screenshot

        cogs = cog_detector.apply(screenshot)
        contours, _ = cv.findContours(cogs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        if time.time() - last_time >= 3 and not looking_at_minimap:
            pg.press('alt')
            looking_at_minimap = True
            last_time = time.time()
            print("start looking at minimap")

        if looking_at_minimap:
            arrow = arrow_detector.apply(screenshot)
            try:
                try:
                    pos, direction = find_arrow_direction(arrow)
                except Exception as e:
                    print(f"arrow_detection_error: {e}")
                if not path_calculated:
                    path = calculate_path(minimap_graph, pos, destination_node)
                    path_calculated = True
                    path_viz = draw_path_on_image(minimap_graph, path)
                    target_node = 0
                    target_pos = minimap_graph.nodes[path[target_node]]['pos']
                    print(f"path={path}")

                angle = calc_angle_to_targ(pos, direction, target_pos)
                if np.abs(angle) > np.pi / 2:
                    pg.keyUp('up')
                else:
                    pg.keyDown('up')
                if angle < -angle_thresh / 2:
                    pg.keyDown('left')
                    pg.keyUp('right')
                elif angle > angle_thresh / 2:
                    pg.keyDown('right')
                    pg.keyUp('left')
                else:
                    pg.keyUp('left')
                    pg.keyUp('right')
                    if time.time() - last_time >= 2:
                        print("quit looking at minimap")
                        pg.press('alt')
                        looking_at_minimap = False
                        last_time = time.time()

                # update target node logic
                if distance(pos, target_pos) < node_reached_thresh:
                    target_node += 1
                    target_pos = minimap_graph.nodes[path[target_node]]['pos']
                    print(f"new_target_node={path[target_node]}")
            except Exception as e:
                print(f"exception: {e}")
            
        # calculate `danger`
        try:
            largest = contours[0]
            largest_area = cv.contourArea(largest)
            cog_x, cog_y = calc_centroid(largest)
        except:
            largest_area = 0
            cog_x = 0

        diff_x = cog_x - facing_x # if negative turn right
        if largest_area > 1800 and np.abs(diff_x) < 180:
            new_danger_status = True
        else:
            new_danger_status = False
        if new_danger_status != danger:
            danger = new_danger_status
            print(f"danger={danger}")

        #cog avoid logic
        if not danger:
            if not looking_at_minimap:
                walkable = walkable_detector.apply(screenshot)
                target_x, _ = mean_coord(walkable)
                if target_x < facing_x - 50:
                    pg.keyDown('left')
                    pg.keyUp('right')
                elif target_x > facing_x + 50:
                    pg.keyDown('right')
                    pg.keyUp('left')
                else:
                    pg.keyUp('left')
                    pg.keyUp('right')
        else:
            if diff_x <= 0:
                pg.keyDown('right')
                pg.keyUp('left')
            else:
                pg.keyDown('left')
                pg.keyUp('right')

        minimap_viz = path_viz.copy() if path_viz is not None else np.zeros((481, 640, 3))
        try:
            color=(255, 0, 255)
            thickness=2
            k = 10
            endpoint = (pos[0] + int(k*direction[0]), pos[1] + int(k*direction[1]))
            cv.arrowedLine(minimap_viz, pos, endpoint, color, thickness)
        except:
            pass
        try:
            viz = visualise([cogs, walkable])
            cv.imshow('visualisation', viz)
        except:
            pass
        cv.imshow('minimap', minimap_viz)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

def calc_angle_to_target(pos, direction, target_pos):
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)
    
    def angle_between(v1, v2):
        return np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
    
    direction_to_targ = (target_pos[0] - pos[0], target_pos[1] - pos[1])

    return angle_between(direction_to_targ, direction)

def calc_angle_to_targ(player_pos, player_dir, target_pos):
    """
    Calculate the angle from the player's direction to the target.

    Parameters:
    player_pos (tuple): The (x, y) position of the player.
    player_dir (tuple): The (x, y) direction vector of the player.
    target_pos (tuple): The (x, y) position of the target.

    Returns:
    float: The angle in degrees from the player's direction to the target.
           The range is between -180 and 180, where negative indicates a left turn.
    """
    # Convert tuples to numpy arrays for vector operations
    player_pos = np.array(player_pos)
    player_dir = np.array(player_dir)
    target_pos = np.array(target_pos)

    # Calculate the normalized direction vectors
    target_dir = target_pos - player_pos
    target_dir_normalized = target_dir / np.linalg.norm(target_dir)
    player_dir_normalized = player_dir / np.linalg.norm(player_dir)

    # Calculate the angle using the dot product and cross product (or determinant)
    angle_cos = np.dot(player_dir_normalized, target_dir_normalized)
    angle_sin = np.cross(player_dir_normalized, target_dir_normalized)

    # Calculate the angle in radians, and then convert to degrees
    angle = np.arctan2(angle_sin, angle_cos)

    return angle

def load_graph_to_networkx(file_path, graph_name):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if graph_name not in data:
        print(f"Graph '{graph_name}' not found in the file.")
        return None

    G = nx.Graph()
    nodes = data[graph_name]['nodes']
    edges = data[graph_name]['edges']

    for node in nodes:
        G.add_node(node['id'], pos=(node['x'], node['y']))

    for edge in edges:
        G.add_edge(edge['start'], edge['end'], weight=edge['distance'])

    return G

def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_path(G, start_point, final_node):
    if G is None:
        print("Graph is not loaded.")
        return []

    # Find the nearest node to the start point
    nearest_node = None
    min_distance = float('inf')
    for node, data in G.nodes(data=True):
        dist = distance(start_point, data['pos'])
        if dist < min_distance:
            nearest_node = node
            min_distance = dist

    if nearest_node is None:
        print("No nearest node found.")
        return []

    # Calculate shortest path
    try:
        path = nx.shortest_path(G, source=nearest_node, target=final_node, weight='weight')
        return path
    except nx.NetworkXNoPath:
        print("No path found.")
        return []

def detect_arrow(filtered, prev_direction, prev_pos):
    arrow_pixels = np.argwhere(filtered == 255)
    # if len(arrow_pixels) == 0:
    #     return prev_pos, prev_direction
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

def find_arrow_direction(binary_image):
    """
    Find the centroid and direction of an arrow in a binary image.

    Parameters:
    binary_image (numpy.ndarray): A binary image containing an arrow.

    Returns:
    tuple: A tuple containing the centroid (x, y) and the direction vector.
    """
    # Ensure the image is binary
    assert set(np.unique(binary_image)).issubset({0, 255}), "Image must be binary"

    # Find coordinates of all foreground pixels
    y, x = np.where(binary_image == 255)
    points = np.column_stack((x, y)).astype(np.float32)

    # Perform PCA
    mean, eigenvectors = cv.PCACompute(points, mean=None)

    # Calculate the unit vector of the principal axis
    principal_axis = eigenvectors[0]

    # Project points onto the principal axis
    projections = np.dot(points - mean, principal_axis)

    # Find the halfway point along the principal component
    min_proj, max_proj = np.min(projections), np.max(projections)
    halfway_proj = (min_proj + max_proj) / 2

    # Count the number of pixels above and below the halfway point
    above_halfway = np.sum(projections > halfway_proj)
    below_halfway = np.sum(projections < halfway_proj)

    # Determine the direction based on the pixel counts
    direction = principal_axis if above_halfway >= below_halfway else -principal_axis
    #direction = (int(direction[0]), int(direction[1]))

    # Return the centroid and direction
    centroid = (int(mean[0,0]), int(mean[0,1]))
    return centroid, tuple(direction)

def draw_path_on_image(G, path, image_size=(481, 640, 3)):
    # Create a blank image
    img = np.zeros(image_size, np.uint8)

    if not path or G is None:
        print("No path or graph provided.")
        return img

    # Draw the path
    for i in range(len(path) - 1):
        node_start = path[i]
        node_end = path[i + 1]

        if node_start not in G or node_end not in G:
            print(f"Node {node_start} or {node_end} not in the graph.")
            continue

        start_pos = G.nodes[node_start]['pos']
        end_pos = G.nodes[node_end]['pos']

        # Draw line for the edge
        cv.line(img, start_pos, end_pos, (0, 255, 0), 2)

        # Draw circles for the nodes
        cv.circle(img, start_pos, 5, (255, 0, 0), -1)
        cv.circle(img, end_pos, 5, (255, 0, 0), -1)

        # Draw node numbers
        cv.putText(img, str(node_start), (start_pos[0] + 10, start_pos[1] + 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(img, str(node_end), (end_pos[0] + 10, end_pos[1] + 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

    return img

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