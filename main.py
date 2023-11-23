import cv2 as cv
import numpy as np
from window_capture import WindowCapture
from time import sleep
from apply_filter import ApplyFilter
import pyautogui as pg
import json
import networkx as nx
from math import sqrt

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

    # minimap related
    pos, direction = np.array([0,0]), np.array([1,0])
    minimap_graph = load_graph_to_networkx('graphs.json', "ttc-punchlineplace")
    path_calculated = False
    path_viz = None
    angle_thresh = (1/6) * np.pi # if angle below (this/2) turn left
    node_reached_thresh = 10
    target_node = None

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

        arrow = arrow_detector.apply(screenshot)
        pos, direction = detect_arrow(arrow, direction)
        if not path_calculated:
            path = calculate_path(minimap_graph, pos, 9)
            path_calculated = True
            path_viz = draw_path_on_image(minimap_graph, path)
            target_node = 0
            target_pos = minimap_graph.nodes[path[target_node]]['pos']
            print(f"path={path}")

        # update target node logic
        if distance(pos, target_pos) < node_reached_thresh:
            target_node += 1
            target_pos = minimap_graph.nodes[path[target_node]]['pos']
        
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

        print(f"danger={danger}")

        #danger = False

        # turning logic
        if not danger:
            angle = calc_angle_to_target(pos, direction, target_pos)
            #print(f"angle_to_target={angle}")
            if angle < -angle_thresh / 2:
                pg.keyDown('left')
                pg.keyUp('right')
            elif angle > angle_thresh / 2:
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

        minimap_viz = path_viz.copy()
        color=(255, 0, 255)
        thickness=2
        endpoint = (1*pos[0] + 1*direction[0], 1*pos[1] + 1*direction[1])
        cv.arrowedLine(minimap_viz, pos, endpoint, color, thickness)
        cv.imshow('cogs', cogs)
        cv.imshow('minimap', minimap_viz)
        key = cv.waitKey(1)
        if key == ord('q'):
            wincap.stop()
            cv.destroyAllWindows()
            break

def calc_angle_to_target(pos, direction, target_pos):
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    def angle_between_old(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def angle_between(v1, v2):
        return np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
    
    direction_to_targ = (target_pos[0] - pos[0], target_pos[1] - pos[1])

    return angle_between(direction_to_targ, direction)

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