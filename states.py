from apply_filter import ApplyFilter
import numpy as np
import cv2 as cv
import time
import json
import networkx as nx
from utils import Utils

class State:
    def __init__(self, context, bot):
        self.context = context
        self.bot = bot

    def enter(self):
        pass

    def update(self, screenshot):
        pass

    def exit(self):
        pass

class StreetNavigation(State):
    def __init__(self, bot, street):
        super().__init__(context=None, bot=bot)
        self.street = street
        self.state_objs = {
            'walking': Walking(self, bot),
            'checking_minimap': CheckingMinimap(self, bot, street)
        }
        self.state = None
        self.prev_state = None

        self.cog_detector = ApplyFilter('cogs')
        self.dangerous_cog_size = 1800
        self.dangerous_cog_angle = 180
        self.in_danger = False

        self.open_map_every = 3
        self.last_time = time.time()

    def set_state(self, new_state_name):
        if self.state:
            self.state.exit()
        self.prev_state = self.state
        self.state = self.state_objs[new_state_name]
        self.state.enter()

    def enter(self):
        self.set_state('checking_minimap')

    def update(self, screenshot):
        self.avoid_cogs(screenshot)
        
        self.should_check_minimap()

        print(f"in_danger={self.in_danger}, state={self.state.__class__.__name__}")

        if not self.in_danger:
            # Call update on the current state
            if self.state:
                self.state.update(screenshot)

    def avoid_cogs(self, screenshot):
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
            self.in_danger = True
        else:
            self.in_danger = False
        
        if self.in_danger:
            if diff_x <= 0:
                self.bot.turn_right()
            else:
                self.bot.turn_left()
        
    def should_check_minimap(self):
        if self.state != self.state_objs['checking_minimap']:
            if time.time() - self.last_time >= self.open_map_every:
                self.set_state('checking_minimap')
                self.last_time = time.time()
                print("start looking at minimap")

class Walking(State):
    def __init__(self, context, bot):
        super().__init__(context, bot)

        self.walkable_det = ApplyFilter('ttc-street-walkable')
        self.walkable_x = self.bot.facing_x
        self.walkable_turn_thresh = 50

    def update(self, screenshot):
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

class CheckingMinimap(State):
    def __init__(self, context, bot, street):
        super().__init__(context, bot)

        self.minimap_dets = {'arrow': ApplyFilter('arrow'),
                             'taskicon': ApplyFilter('taskicon'),
                             'minimap-mask': ApplyFilter('minimap-mask')}
        self.minimap_graph = self.load_minimap_graph(street)
        self.minimap_mask = None
        self.minimap_path = None
        self.keep_map_open_for = 1.3 # seconds
        self.node_angle_thresh = (1/12) * np.pi # if angle below (-thresh/2) turn left
        self.node_reached_thresh = 10 # pixels
        self.next_node_idx = 0
        self.arrow_pos, self.arrow_dir = None, None

    def enter(self):
        self.bot.toggle_minimap()

    # def exit(self):
    #     self.bot.toggle_minimap()
    
    def update(self, screenshot):
        # Initialise minimap mask
        if self.minimap_mask is None:
            if time.time() - self.context.last_time > 1:
                self.minimap_mask = self.minimap_dets['minimap-mask'].apply(screenshot)
                self.context.last_time = time.time()
                self.bot.start_moving()
            return

        # Calc arrow vector
        arrow = self.detect_on_minimap('arrow', screenshot)
        # If no arrow detected this frame
        if not np.count_nonzero(arrow):
            return
        self.arrow_pos, self.arrow_dir = self.calc_arrow_vector(arrow)

        # Initialise minimap path
        if self.minimap_path is None:
            taskicon = self.detect_on_minimap('taskicon', screenshot)
            if not np.count_nonzero(taskicon):
                return
            self.task_location = Utils.calc_centroid(taskicon)
            self.calc_minimap_path()
            #self.path_viz = Utils.draw_path_on_image(self.minimap_graph, self.minimap_path, self.task_location)

        next_node_pos = self.get_node_pos(self.next_node_idx)
        # Angle between arrow direction and direction to next node
        angle = self.calc_angle_to_targ(next_node_pos)
        print(f"angle={angle}")

        if np.abs(angle) > np.pi / 2:
            self.bot.stop_moving()
        else:
            self.bot.start_moving()
        
        if angle < -self.node_angle_thresh / 2:
            self.bot.turn_left()
        elif angle > self.node_angle_thresh / 2:
            self.bot.turn_right()
        else:
            self.bot.stop_turning()
            # Calculate if minimap should be closed
            if time.time() - self.context.last_time >= self.keep_map_open_for:
                print("quit looking at minimap")
                self.bot.toggle_minimap()
                self.context.last_time = time.time()
                self.context.set_state('walking')

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

    def detect_on_minimap(self, item_name, screenshot):
        item = np.zeros_like(screenshot[:,:,0])

        items = self.minimap_dets[item_name].apply(screenshot)

        # Find contours in the arrow-detected image
        contours, _ = cv.findContours(items, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Check if any point of the contour is within the mask
            for point in contour:
                if self.minimap_mask[point[0][1], point[0][0]] == 255:
                    # Draw the full contour of the arrow
                    cv.drawContours(item, [contour], -1, (255, 255, 255), -1)
                    break

        return item
    
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
        
    def get_node_pos(self, node_idx):
        return self.minimap_graph.nodes[self.minimap_path[node_idx]]['pos']