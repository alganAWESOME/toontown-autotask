import numpy as np
import cv2 as cv
import json
import networkx as nx

class Visualizer:
    def __init__(self) -> None:
        self.map = cv.imread('filter-loopylane.png')
        self.graph = self.load_graph()
        self.draw_graph_on_map()
        self.map_copy = self.map.copy()

    def load_graph(self):
        with open("graph_data.json", "r") as file:
            data = json.load(file)

        G = nx.Graph()
        for node in data["nodes"]:
            G.add_node(node["id"], x=node["x"], y=node["y"])
        for edge in data["edges"]:
            G.add_edge(edge["start"], edge["end"], weight=edge["distance"])

        return G

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

        return self.map