import numpy as np
import cv2 as cv
import json
import networkx as nx

class Pathfinder:
    def __init__(self) -> None:
        self.map_white = cv.imread('filter-loopylane.png')
        self.map_white_copy = self.map_white.copy()
        self.graph = self.load_graph()

    def load_graph(self):
        with open("graph_data.json", "r") as file:
            data = json.load(file)

        G = nx.Graph()

        for node in data["nodes"]:
            G.add_node(node["id"], x=node["x"], y=node["y"])

        for edge in data["edges"]:
            G.add_edge(edge["start"], edge["end"], weight=edge["distance"])

        return G

    def visualize(self, pos, direction):
        self.map_white = self.map_white_copy.copy()

        radius, color, thickness = 3, (0,255,0), -1

        cv.circle(self.map_white, pos, radius, color, thickness)
        cv.arrowedLine(self.map_white,pos, pos+direction,(255,0,0),3)

        print(f"Direction {direction}")
        print(f"Current position {pos}")

        return self.map_white