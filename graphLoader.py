import numpy as np
import networkx as nx

class graphml:    
    """imports from a graphML file rapresenting the adjacency matrix"""
    graph = {}
    vertices = []

    def __init__(self, filename):
        self.graph = nx.read_graphml(filename, node_type= int, edge_key_type=int)
        print(self.graph)

        # get positions of nodes
        for n in self.graph.nodes():
            self.vertices.append([self.graph.node[n]['x'], self.graph.node[n]['y']])
        
        # normalize positions
        max = 1
        for coord in self.vertices:
            if abs(coord[0])>max:
                max = abs(coord[0])
            if abs(coord[1])>max:
                max = abs(coord[1])
        for indexCoord in range(len(self.vertices)):
            self.vertices[indexCoord] = [self.vertices[indexCoord][0]/max, self.vertices[indexCoord][1]/max]

        # add length attibute to every edge
        for n, nbrs in self.graph.adj.items():
            for nbr, _ in nbrs.items():
                length = np.linalg.norm(np.subtract(self.vertices[n], self.vertices[nbr]), 2)
                self.graph[n][nbr]['distance'] = length

