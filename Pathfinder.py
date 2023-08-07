import numpy as np
import cv2 as cv
import json
import networkx as nx
import pyautogui as pg

class Pathfinder:
    def __init__(self) -> None:
        self.map = cv.imread('filter-loopylane.png')
        self.graph = self.load_graph()
        self.draw_graph_on_map()
        self.map_copy = self.map.copy()
        self.path = None
        self.idx = 0
        self.node_reached_threshold = 10

    def main(self, arrow_pos, arrow_direction):
        # If path not initialised, then do so
        # May need to modify later so that self.path is initialised at __init__
        if self.path is None:
            self.path = self.find_shortest_path(arrow_pos)
        else:
            # while self.path is not empty
            # always hold forward
            # find direction between current position and next node
            # hold right or left depending on arrow direction
            print("we're here")
            if self.idx <= len(self.path):
                pg.press('up')
                next_node = self.path[self.idx]
                #print(f"next node {self.graph[next_node]}")
                next_node_pos = np.array([self.graph.nodes[next_node]['x'], self.graph.nodes[next_node]['y']])
                if self.distance(arrow_pos, next_node) > self.node_reached_threshold:
                    desired_direction = next_node_pos - arrow_pos
                    if self.angle(arrow_direction, desired_direction) > 0:
                        pg.press('left')
                    else:
                        pg.press('right')
                else:
                    self.idx += 1
            else:
                print("Done?")
        
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
        # Modify above code to incorporate step 2, for now target is node 4

        # Step 3) Calculate shortest path
        shortest_path = nx.shortest_path(self.graph, node_nearest_to_player, 4, weight='weight')
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
    
    def draw_graph_on_map(self):
        # Font settings for node labels
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (0, 0, 0)

        # Get the node positions from the graph and draw them on the image
        for node in self.graph.nodes():
            x, y = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
            cv.circle(self.map, (x, y), 5, (0, 0, 255), -1)  # Node as a red circle
            cv.putText(self.map, str(node), (x + 8, y), font, font_scale, font_color, font_thickness, lineType=cv.LINE_AA)

        # Draw the edges on the image
        for edge in self.graph.edges():
            x1, y1 = self.graph.nodes[edge[0]]['x'], self.graph.nodes[edge[0]]['y']
            x2, y2 = self.graph.nodes[edge[1]]['x'], self.graph.nodes[edge[1]]['y']
            cv.line(self.map, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Edge as a blue line

    def visualize(self, pos, direction):
        self.map = self.map_copy.copy()

        radius, color, thickness = 3, (0,255,0), -1

        cv.circle(self.map, pos, radius, color, thickness)
        cv.arrowedLine(self.map,pos, pos+direction,(255,0,0),3)

        # print(f"Direction {direction}")
        # print(f"Current position {pos}")

        return self.map