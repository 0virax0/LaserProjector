import csv
import numpy as np
import networkx as nx

class graphImporter:    
    """imports from a graphML file rapresenting the adjacency matrix"""
    weightedGraph = {}

    def __init__(self, filename):
        self.weightedGraph = nx.read_graphml(filename, node_type= int, edge_key_type=<class 'int'>)

g = graphImporter("graph.csv")
