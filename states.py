from apply_filter import ApplyFilter
import numpy as np
import cv2 as cv
import time
import json
import networkx as nx
from utils import Utils

class StreetNavigation:
    def __init__(self, bot) -> None:
        self.bot = bot
        self.walkable_det = ApplyFilter('ttc-street-walkable')
        self.cog_detector = ApplyFilter('cogs')
        self.arrow_det = ApplyFilter('sillystreet-arrow')
        self.visualizations = {}

        self.state = "NAVIGATING_STREET"

        # Minimap related
        self.destination_node = 0
        self.looking_at_minimap = True
        self.open_map_every = 3 #seconds
        self.keep_map_open_for = 1.3
        self.minimap_graph = self.load_minimap_graph('sillystreet')
        self.minimap_path = None
        self.angle_thresh = (1/12) * np.pi # if angle below (-this/2) turn left
        self.node_reached_thresh = 10 #pixels
        self.next_node_idx = 0
        self.last_time = time.time()
        self.arrow_pos, self.arrow_dir = None, None

        # Cog avoidance related
        self.in_danger = False
        self.dangerous_cog_size = 1800
        self.dangerous_cog_angle = 180

        # Walkable related
        self.walkable_x = self.bot.facing_x
        self.walkable_turn_thresh = 50

        self.enter()

    def enter(self):
        self.bot.toggle_minimap()
        self.bot.start_moving()
        
    def update(self, screenshot):
        # Calculate `looking_at_minimap` status
        if time.time() - self.last_time >= self.open_map_every and not self.looking_at_minimap:
            self.bot.toggle_minimap()
            self.looking_at_minimap = True
            self.last_time = time.time()
            print("start looking at minimap")
        
        if self.looking_at_minimap:
            self.execute_minimap_logic(screenshot)

        # Calculate danger status
        cogs = self.cog_detector.apply(screenshot)
        cog_contours, _ = cv.findContours(cogs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cog_contours = sorted(cog_contours, key=cv.contourArea, reverse=True)

        if cog_contours: # If there are any cog contours at all
            closest_cog = cog_contours[0]
            closest_cog_size = cv.contourArea(closest_cog)
            cog_x, _ = Utils.calc_centroid(closest_cog)
        else:
            closest_cog = None
            closest_cog_size = 0
            cog_x = 0

        diff_x = cog_x - self.bot.facing_x # If negative, turn right
        if closest_cog_size > self.dangerous_cog_size and diff_x < self.dangerous_cog_angle:
            if not self.in_danger:
                print("in_danger=True")
            self.in_danger = True
        else:
            if self.in_danger:
                print("in_danger=False")
            self.in_danger = False

        walkable = np.zeros_like(screenshot[:,:,2])
        if not self.in_danger:
            # If not looking_at_minimap, follow mean walkable point
            if not self.looking_at_minimap:
                walkable = self.walkable_det.apply(screenshot)
                if np.count_nonzero(walkable):
                    walkable_x, _ = Utils.calc_centroid(walkable)
                else:
                    walkable_x = self.bot.facing_x
                
                if walkable_x < self.bot.facing_x - self.walkable_turn_thresh:
                    self.bot.turn_left()
                elif walkable_x > self.bot.facing_x + self.walkable_turn_thresh:
                    self.bot.turn_right()
                else:
                    self.bot.stop_turning()
        else:
            if diff_x <= 0:
                self.bot.turn_right()
            else:
                self.bot.turn_left()

        viz = Utils.visualise([cogs, walkable])
        self.visualizations['Vision'] = viz

    def execute_minimap_logic(self, screenshot):
        arrow = self.arrow_det.apply(screenshot)

        try:
            self.arrow_pos, self.arrow_dir = self.calc_arrow_vector(arrow)
        except:
            pass

        if self.arrow_pos is None or self.arrow_dir is None:
            return
        
        if self.minimap_path is None:
            task_location_det = ApplyFilter('sillystreet-task')
            task_logo = task_location_det.apply(screenshot)
            if not np.count_nonzero(task_logo):
                return
            self.task_location = Utils.calc_centroid(task_logo)
            self.calc_minimap_path()

            self.path_viz = Utils.draw_path_on_image(self.minimap_graph, self.minimap_path, self.task_location)

        minimap_viz = Utils.draw_arrow_on_image(self.path_viz, self.arrow_pos, self.arrow_dir)
        self.visualizations['Minimap'] = minimap_viz

        next_node_pos = self.get_node_pos(self.next_node_idx)
        # Angle between arrow direction and direction to next node
        angle = self.calc_angle_to_targ(next_node_pos)
        print(f"angle={angle}")

        if np.abs(angle) > np.pi / 2:
            self.bot.stop_moving()
        else:
            self.bot.start_moving()
        
        if angle < -self.angle_thresh / 2:
            self.bot.turn_left()
        elif angle > self.angle_thresh / 2:
            self.bot.turn_right()
        else:
            self.bot.stop_turning()
            # Calculate if minimap should be closed
            if time.time() - self.last_time >= self.keep_map_open_for:
                print("quit looking at minimap")
                self.bot.toggle_minimap()
                self.looking_at_minimap = False
                self.last_time = time.time()

        # Update next_node
        if Utils.euclidean_dist(self.arrow_pos, next_node_pos) < self.node_reached_thresh:
            # If we have not reached final node in path
            if self.next_node_idx != len(self.minimap_path)-1:
                self.next_node_idx += 1
                next_node_pos = self.get_node_pos(self.next_node_idx)
                print(f"new_target_node={self.minimap_path[self.next_node_idx]}")
            else:
                next_node_pos = self.task_location
                self.navigating_to_door = True
                print("moving towards task")
    
    @staticmethod    
    def calc_arrow_vector(binary_image):
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

        # Return the centroid and direction
        centroid = (int(mean[0,0]), int(mean[0,1]))
        return centroid, tuple(direction)
    
    def calc_angle_to_targ(self, target_pos):
        # Convert tuples to numpy arrays for vector operations
        player_pos = np.array(self.arrow_pos)
        player_dir = np.array(self.arrow_dir)
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
    
    def get_node_pos(self, node_idx):
        return self.minimap_graph.nodes[self.minimap_path[node_idx]]['pos']

    def calc_minimap_path(self):
        # Find the nearest node to the start point
        start_nearest_node = None
        start_min_distance = float('inf')
        for node, data in self.minimap_graph.nodes(data=True):
            dist = Utils.manhattan_dist(self.arrow_pos, data['pos'])
            if dist < start_min_distance:
                start_nearest_node = node
                start_min_distance = dist

        # Find the nearest node to the final destination
        end_nearest_node = None
        end_min_distance = float('inf')
        for node, data in self.minimap_graph.nodes(data=True):
            dist = Utils.manhattan_dist(self.task_location, data['pos'])
            if dist < end_min_distance:
                end_nearest_node = node
                end_min_distance = dist

        # Compute the shortest path from the start node to the end node
        self.minimap_path = nx.shortest_path(self.minimap_graph,
                                            source=start_nearest_node,
                                            target=end_nearest_node,
                                            weight='weight')
    
    @staticmethod
    def load_minimap_graph(graph_name):
        with open('graphs.json', 'r') as file:
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
