import csv
import numpy as np
import networkx as nx

class graphImporter:    
    """imports from a graphML file rapresenting the adjacency matrix"""
    weightedGraph = {}
    vertices = []

    def __init__(self, filename):
        self.weightedGraph = nx.read_graphml(filename, node_type= int, edge_key_type=int)
        print(self.weightedGraph)

        # get positions of nodes
        for n in self.weightedGraph.nodes():
            self.vertices.append([self.weightedGraph.node[n]['x'], self.weightedGraph.node[n]['y']])
        
        # normalize positions

g = graphImporter("graph.graphml")
