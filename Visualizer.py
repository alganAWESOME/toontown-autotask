import numpy as np
import cv2 as cv
import json
import networkx as nx
import keyboard

class Visualizer:
    def __init__(self) -> None:
        self.map_filename = 'filter-loopylane.png'
        self.map = cv.imread(self.map_filename)
        self.map_cleared = self.map.copy()
        self.map_copy = self.map.copy()
        self.graph = self.load_graph()
        self.draw_graph_on_map()

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
        self.clear_graph_from_map()

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

        self.map_copy = self.map.copy()

    def clear_graph_from_map(self):
        self.map = self.map_cleared.copy()
        self.map_copy = self.map.copy()

    def visualize(self, pos, direction):
        self.map = self.map_copy.copy()

        radius, color, thickness = 3, (0,255,0), -1

        cv.circle(self.map, pos, radius, color, thickness)
        cv.arrowedLine(self.map,pos, pos+direction,(255,0,0),3)

        return self.map

class GraphCreator:
    def __init__(self, visualizer) -> None:
        self.visualizer = visualizer
        self.visualizer.clear_graph_from_map()

        self.current_node_id = 0
        self.new_graph = nx.Graph()
        self.new_node_minimum_distance = 10

        self.keys_held = set()
        self.keys = ['shift', 'backspace', 'enter']

    def main(self, pos):
        for key in self.keys:
            if keyboard.is_pressed(key):
                if key not in self.keys_held:
                    self.keys_held.add(key)
                    self.graph_creator(pos, key)
            else:
                self.keys_held.discard(key)

    def graph_creator(self, pos, key):
        if key=='shift':
            if self.current_node_id == 0:
                self.new_graph.add_node(self.current_node_id, x=pos[0], y=pos[1])
                self.current_node_id += 1
            else:
                distance = self.distance(pos, self.current_node_id-1)
                if distance > self.new_node_minimum_distance:
                    self.new_graph.add_node(self.current_node_id, x=pos[0], y=pos[1])
                    self.new_graph.add_edge(self.current_node_id, self.current_node_id-1, distance=distance)
                    self.current_node_id += 1
            self.update_visualizer()

        if key=='backspace':
            if self.current_node_id > 0:
                self.current_node_id -= 1
                self.new_graph.remove_node(self.current_node_id)
                self.update_visualizer()
        
        if key=='enter':
            self.save_graph_to_json()

    def distance(self, pos, node):
        dist_squared = (pos[0] - self.new_graph.nodes[node]['x'])**2 + (pos[1] - self.new_graph.nodes[node]['y'])**2
        return np.sqrt(dist_squared)
                
    def update_visualizer(self):
        self.visualizer.graph = self.new_graph
        self.visualizer.draw_graph_on_map()
    
    def save_graph_to_json(self):
        nodes_data = [{"id": int(n), "x": int(self.new_graph.nodes[n]['x']), "y": int(self.new_graph.nodes[n]['y'])} for n in self.new_graph.nodes()]
        edges_data = [{"start": u, "end": v, "distance": self.new_graph[u][v]['distance']} for u, v in self.new_graph.edges()]
        graph_data = {"nodes": nodes_data, "edges": edges_data}
        
        with open('graph_data.json', 'w') as f:
            json.dump(graph_data, f)

        print(f"Saved {graph_data}")