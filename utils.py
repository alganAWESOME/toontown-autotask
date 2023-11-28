import numpy as np
import math
import cv2 as cv

class Utils:
    @staticmethod
    def manhattan_dist(p1, p2):
        return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])
    
    @staticmethod
    def euclidean_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def calc_centroid(contour):
        M = cv.moments(contour)
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    
    @staticmethod
    def visualise(binary_images):
        colors = [(0,0,255), (255,0,0), (0, 255, 0), (0, 128, 255), (0,255,255), (0,255,128)]
        viz = np.zeros_like(cv.cvtColor(binary_images[0], cv.COLOR_GRAY2BGR))
        for i, img in enumerate(binary_images):
            viz[img != 0] = colors[i]
        return viz

    @staticmethod
    def draw_path_on_image(G, path, task_location, image_size=(481, 640, 3)):
        # Create a blank image
        img = np.zeros(image_size, np.uint8)

        if not path or G is None:
            print("No path or graph provided.")
            return img

        # Draw the path
        for i in range(len(path)):
            node_start = path[i]
            if i < len(path)-1:
                node_end = path[i + 1]

                if node_start not in G or node_end not in G:
                    print(f"Node {node_start} or {node_end} not in the graph.")
                    continue

                end_pos = G.nodes[node_end]['pos']
            else:
                end_pos = task_location
            start_pos = G.nodes[node_start]['pos']

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
    
    @staticmethod
    def draw_arrow_on_image(img, arrow_pos, arrow_dir):
        image = img.copy()
        color=(255, 0, 255)
        thickness=2
        k = 10
        endpoint = (arrow_pos[0] + int(k*arrow_dir[0]), arrow_pos[1] + int(k*arrow_dir[1]))
        cv.arrowedLine(image, arrow_pos, endpoint, color, thickness)

        return image