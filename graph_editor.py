import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import json
import math

class Node:
    def __init__(self, x, y, stop_required=False):
        self.x = x
        self.y = y
        self.stop_required = stop_required
        self.connections = []

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Pathfinding Tool")
        self.geometry("800x600")

        self.nodes = []
        self.temp_node = None  # Temporary node when making connections

        # Canvas for the minimap image
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.canvas.bind("<Button-2>", self.on_canvas_middle_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

        # Loading minimap image
        self.image_path = "map-loopylane.png"
        self.load_image(self.image_path)

        # Button to save the graph to JSON
        self.save_button = tk.Button(self, text="Save to JSON", command=self.save_to_json)
        self.save_button.pack(side=tk.BOTTOM)

        # Button to load existing graph from JSON
        self.load_button = tk.Button(self, text="Load Graph", command=self.load_from_json)
        self.load_button.pack(side=tk.BOTTOM)


    def load_image(self, image_path):
        img = Image.open(image_path)
        self.imgtk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_canvas_left_click(self, event):
        for node in self.nodes:
            radius = 10
            if abs(node.x - event.x) <= radius and abs(node.y - event.y) <= radius:
                if self.temp_node:  # If temp_node exists, we are making a connection
                    # Check if a connection between these two nodes already exists
                    if node not in self.temp_node.connections:
                        node.connections.append(self.temp_node)
                        self.temp_node.connections.append(node)
                        self.canvas.create_line(self.temp_node.x, self.temp_node.y, node.x, node.y)
                    self.temp_node = None
                    return
                else:
                    # Otherwise, we set the current node as temp_node to make a connection
                    self.temp_node = node
                    return

        # If we're not clicking on an existing node and no connection is in progress,
        # create a new node
        if not self.temp_node:
            node = Node(event.x, event.y)
            color = 'red' if node.stop_required else 'blue'
            self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill=color)
            self.nodes.append(node)

    def on_canvas_middle_click(self, event):
        for node in self.nodes:
            radius = 10
            if abs(node.x - event.x) <= radius and abs(node.y - event.y) <= radius:
                node.stop_required = not node.stop_required  # Toggle stop_required
                color = 'red' if node.stop_required else 'blue'
                self.canvas.create_oval(node.x - 5, node.y - 5, node.x + 5, node.y + 5, fill=color, outline='')
                return


    def on_canvas_right_click(self, event):
        for node in self.nodes:
            if abs(node.x - event.x) <= 5 and abs(node.y - event.y) <= 5:
                for conn in node.connections:
                    conn.connections.remove(node)
                    # You may also want to delete the line visually from the canvas.
                self.nodes.remove(node)
                self.redraw_canvas()
                return

    def save_to_json(self):
        graph_data = {'nodes': [], 'edges': []}
        for idx, node in enumerate(self.nodes):
            graph_data['nodes'].append({
                'id': idx,
                'x': node.x,
                'y': node.y,
                'stop_required': node.stop_required
            })


            for conn in node.connections:
                conn_idx = self.nodes.index(conn)
                # To avoid duplicate edges, we'll ensure that start < end.
                start, end = min(idx, conn_idx), max(idx, conn_idx)
                edge_data = {
                    'start': start,
                    'end': end,
                    'distance': node.distance(conn)
                }
                if edge_data not in graph_data['edges']:
                    graph_data['edges'].append(edge_data)

        with open("graph_data.json", "w") as outfile:
            json.dump(graph_data, outfile)

    def redraw_canvas(self):
        self.canvas.delete(tk.ALL)
        self.load_image(self.image_path)
        for node in self.nodes:
            self.canvas.create_oval(node.x - 5, node.y - 5, node.x + 5, node.y + 5, fill='blue')
            for conn in node.connections:
                self.canvas.create_line(node.x, node.y, conn.x, conn.y)

    def load_from_json(self):
        with open("graph_data.json", "r") as infile:
            graph_data = json.load(infile)

        # Clear the canvas and node list first
        self.canvas.delete(tk.ALL)
        self.load_image(self.image_path)
        self.nodes = []

        # Load nodes
        for node_data in graph_data['nodes']:
            node = Node(node_data['x'], node_data['y'], node_data.get('stop_required', False))
            color = 'red' if node.stop_required else 'blue'
            self.canvas.create_oval(node.x - 5, node.y - 5, node.x + 5, node.y + 5, fill=color)
            self.nodes.append(node)

        # Load edges/connections
        for edge_data in graph_data['edges']:
            start_node = self.nodes[edge_data['start']]
            end_node = self.nodes[edge_data['end']]
            
            if end_node not in start_node.connections:
                start_node.connections.append(end_node)
                end_node.connections.append(start_node)
                
                self.canvas.create_line(start_node.x, start_node.y, end_node.x, end_node.y)


if __name__ == "__main__":
    app = Application()
    app.mainloop()
