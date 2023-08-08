import numpy as np
import cv2 as cv
import json
import networkx as nx
import keyboard

class Visualizer:
    def __init__(self) -> None:
        self.map_filename = 'filter-loopylane.png'
        self.map = cv.imread(self.map_filename)
        self.graph = self.load_graph()
        self.draw_graph_on_map(self.graph)

        # For graph creation mode
        self.current_node_id = 0
        self.new_graph = nx.Graph()
        self.cleared_old_graph = False
        self.new_node_minimum_distance = 10
        self.backspace_already_pressed = False

    def load_graph(self):
        with open("graph_data.json", "r") as file:
            data = json.load(file)

        G = nx.Graph()
        for node in data["nodes"]:
            G.add_node(node["id"], x=node["x"], y=node["y"])
        for edge in data["edges"]:
            G.add_edge(edge["start"], edge["end"], weight=edge["distance"])

        return G

    def draw_graph_on_map(self, graph):
        self.clear_graph_from_map()

        # Font settings for node labels
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (0, 0, 0)

        # Get the node positions from the graph and draw them on the image
        for node in graph.nodes():
            x, y = graph.nodes[node]['x'], graph.nodes[node]['y']
            cv.circle(self.map, (x, y), 5, (0, 0, 255), -1)  # Node as a red circle
            cv.putText(self.map, str(node), (x + 8, y), font, font_scale, font_color, font_thickness, lineType=cv.LINE_AA)

        # Draw the edges on the image
        for edge in graph.edges():
            x1, y1 = graph.nodes[edge[0]]['x'], graph.nodes[edge[0]]['y']
            x2, y2 = graph.nodes[edge[1]]['x'], graph.nodes[edge[1]]['y']
            cv.line(self.map, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Edge as a blue line

        self.map_copy = self.map.copy()

    def clear_graph_from_map(self):
        self.map = cv.imread(self.map_filename)
        self.map_copy = self.map.copy()

    def visualize(self, pos, direction):
        self.map = self.map_copy.copy()

        radius, color, thickness = 3, (0,255,0), -1

        cv.circle(self.map, pos, radius, color, thickness)
        cv.arrowedLine(self.map,pos, pos+direction,(255,0,0),3)

        return self.map

    def graph_creator(self, pos):
        if not self.cleared_old_graph:
            self.clear_graph_from_map()
            self.cleared_old_graph = True
        if keyboard.is_pressed('shift'):
            if self.current_node_id == 0:
                self.new_graph.add_node(self.current_node_id, x=pos[0], y=pos[1])
                self.current_node_id += 1
            else:
                distance = self.distance(pos, self.current_node_id-1)
                if distance > self.new_node_minimum_distance:
                    self.new_graph.add_node(self.current_node_id, x=pos[0], y=pos[1])
                    self.new_graph.add_edge(self.current_node_id, self.current_node_id-1, distance=distance)
                    self.current_node_id += 1
            # self.new_graph.add_node(self.current_node_id, x=pos[0], y=pos[1])
            # # If there already exists at least one node
            # if self.current_node_id != 0:
            #     edge_length = self.distance(pos, self.current_node_id-1)
            #     self.new_graph.add_edge(self.current_node_id, self.current_node_id-1, distance=edge_length)
            # self.current_node_id += 1
            self.draw_graph_on_map(self.new_graph)

        # We need to modify this code so that it works event-based:
        # `keyboard.hook(event_handler)`
        if keyboard.is_pressed('backspace'):
            if not self.backspace_already_pressed:
                if self.current_node_id > 0:
                    self.current_node_id -= 1
                    self.new_graph.remove_node(self.current_node_id)
                    self.draw_graph_on_map(self.new_graph)
                self.backspace_already_pressed = True
        else:
            self.backspace_already_pressed = False
        
        if keyboard.is_pressed('enter'):
            self.save_graph_to_json()

    def distance(self, pos, node):
        dist_squared = (pos[0] - self.new_graph.nodes[node]['x'])**2 + (pos[1] - self.new_graph.nodes[node]['y'])**2
        return np.sqrt(dist_squared)
                
    def save_graph_to_json(self):
        nodes_data = [{"id": int(n), "x": int(self.new_graph.nodes[n]['x']), "y": int(self.new_graph.nodes[n]['y'])} for n in self.new_graph.nodes()]
        edges_data = [{"start": u, "end": v, "distance": self.new_graph[u][v]['distance']} for u, v in self.new_graph.edges()]
        graph_data = {"nodes": nodes_data, "edges": edges_data}
        
        with open('graph_data.json', 'w') as f:
            json.dump(graph_data, f)

        print(f"Saved {graph_data}")