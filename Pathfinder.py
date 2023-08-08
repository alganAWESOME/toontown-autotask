import numpy as np
import cv2 as cv
import json
import networkx as nx
import pyautogui as pg

class Pathfinder:
    def __init__(self, target_node) -> None:
        # self.map = cv.imread('filter-loopylane.png')
        # self.draw_graph_on_map()
        # self.map_copy = self.map.copy()
        self.graph = self.load_graph()
        self.path = None
        self.idx = 0
        self.node_reached_threshold = 10
        self.angle_threshold = 10
        self.angle_sign = None
        self.arrow_key_input = None
        self.target_node = target_node

    def main(self, arrow_pos, arrow_direction):
        # If path not initialised, then do so
        # May need to modify later so that self.path is initialised at __init__
        if self.path is None:
            self.path = self.find_shortest_path(arrow_pos)
            pg.keyDown('up')
        else:
            if self.idx < len(self.path):
                next_node = self.path[self.idx]
                next_node_pos = np.array([self.graph.nodes[next_node]['x'], self.graph.nodes[next_node]['y']])
                # if next node not reached
                if self.distance(arrow_pos, next_node) > self.node_reached_threshold:
                    desired_direction = next_node_pos - arrow_pos
                    angle = self.angle(arrow_direction, desired_direction)
                    # (1) if no direction is being input
                    if self.arrow_key_input is None:
                        self.angle_sign = np.sign(angle)
                        # if bot not facing target node within threshold, make it turn
                        # program jumps to (2)
                        if np.abs(angle) >= self.angle_threshold:
                            if self.angle_sign == 1:
                                self.arrow_key_input = 'right'
                            elif self.angle_sign == -1:
                                self.arrow_key_input = 'left'
                            pg.keyDown(self.arrow_key_input)
                    # (2) if we're already turning
                    else:
                        print(f"Going towards node {next_node}, angle towards node {angle}, should be turning {self.arrow_key_input}")
                        new_angle_sign = np.sign(angle)
                        # check for switch in angle sign, in which case switch arrow key input
                        if new_angle_sign == -self.angle_sign:
                            pg.keyUp(self.arrow_key_input)
                            self.arrow_key_input = self.switch_arrow_key_input(self.arrow_key_input)
                            pg.keyDown(self.arrow_key_input)
                        self.angle_sign = new_angle_sign
                        # if bot direction gets within threshold, stop turning
                        # program jumps to (1)
                        if np.abs(angle) < self.angle_threshold:
                            pg.keyUp(self.arrow_key_input)
                            self.arrow_key_input = None
                else:
                    self.idx += 1
            else:
                pg.keyUp('up')
                print("Done")
    
    @staticmethod
    def switch_arrow_key_input(current_input):
        if current_input == "left":
            return "right"
        elif current_input == "right":
            return "left"
        else:
            raise ValueError("current_input must be either 'right' or 'left'")
        
    def load_graph(self):
        with open("graph_data.json", "r") as file:
            data = json.load(file)

        G = nx.Graph()
        for node in data["nodes"]:
            G.add_node(node["id"], x=node["x"], y=node["y"])
        for edge in data["edges"]:
            G.add_edge(edge["start"], edge["end"], weight=edge["distance"])

        return G
    
    def find_shortest_path(self, pos):
        # Step 1) Find node nearest to player
        node_nearest_to_player = 0
        distance_nearest = self.distance(pos, 0)

        for node in self.graph.nodes():
            distance_current = self.distance(pos, node)
            if distance_current < distance_nearest:
                node_nearest_to_player = node
                distance_nearest = distance_current
        
        print(f"Nearest node to {pos} is {node_nearest_to_player}")
        # Step 2) Find node nearest to end-goal
        # Modify above code to incorporate step 2, for now target is some node

        # Step 3) Calculate shortest path
        shortest_path = nx.shortest_path(self.graph, node_nearest_to_player, self.target_node, weight='weight')
        return shortest_path
    
    def distance(self, pos, node):
        dist_squared = (pos[0] - self.graph.nodes[node]['x'])**2 + (pos[1] - self.graph.nodes[node]['y'])**2
        return np.sqrt(dist_squared)
    
    @staticmethod
    def angle(u, v):
        dot_product = np.dot(u, v)

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        angle_radians = np.arccos(dot_product / (norm_u * norm_v))

        cross_product = np.cross(u, v)

        if cross_product < 0:
            angle_radians = -angle_radians

        angle_degrees = np.degrees(angle_radians)

        return np.int32(angle_degrees)
