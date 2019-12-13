# Input: eulerian networkX graph 
# Output: an eulerian path with minimal angle maximized

# to do that i can consider every possible angle formed between edges on the graph and
# start to delete them starting from the smallest until an eulerian path exists, then 
# i find a path (O(n^3)). 
# To reach O(n^2*log(n)) complexity I only need for each edge to find all largest angles formed 
# between edges connected to the same node. Than i take the smallest and find a path excluding every
# angle < found edge

import numpy as np
import math
import networkx as nx

# to find the largest angle for every edge connected to a node i need a search tree
class Node:
    def __init__(self, key, val):
        self.parent = None
        self.left = None
        self.right = None
        self.key = key
        self.data = val
    
    def insert(self, newKey, val):
        if self.key:
            if newKey < self.key:
                if self.left is None:
                    self.left = Node(newKey,val)
                    self.left.parent = self
                else:
                    self.left.insert(newKey, val)
            else:
                if self.right is None:
                    self.right = Node(newKey, val)
                    self.right.parent = self
                else:
                    self.right.insert(newKey, val)
        else:
            self.key = newKey
            self.data = val

    # Print the tree
    def PrintFrom(self):
        if self.left:
            self.left.PrintFrom()
        print("data:",self.data, "key:", self.key)
        if self.right:
            self.right.PrintFrom()

class AngleSearchTree: 
    def __init__(self):
        self.root = None
        self.minNode = None
        self.maxNode = None

    def insert(self, v):    # v is the vector to insert (x,y) normalized
        newKey = AbsArc(v)  # get AbsArc of the vector normalized
        val = v # data to be saved on the node
        newNode = Node(newKey, val)
        if self.root is None:
            self.root = newNode
            self.minNode = newNode
            self.maxNode = newNode
        else: self.root.insert(newKey, val)

        if self.minNode.key > newKey:
            self.minNode = newNode
        elif self.maxNode.key < newKey:
            self.maxNode = newNode

    def predecessor(self, node):    #cyclic predecessor
        f = node.parent
        while (f is not None) and (f.left is node):
            node = f
            f = f.parent
        
        if f is None:
            return self.maxNode
        return f
    
    def successor(self, node):
        f = node.parent
        while (f is not None) and (f.right is node):
            node = f
            f = f.parent
        
        if f is None:
            return self.minNode
        return f

    # i find the closest value circuiting
    # returns (dot with closest, normalized closest vector)
    def findClosest(self, v):   #v normalized
        # walk tree, if I stop in an empty left tree compare myself to the predecessor
        # specularly to a right tree
        if self.root is None: return None
        key = AbsArc(v)
        curr = self.root

        while curr:
            if curr.key is key:
                return (curr.key, curr.data)
            elif curr.key >= key:
                if curr.left is None:
                    p = self.predecessor(curr)
                    pDot = dot(p.data, v)
                    currDot = dot(curr.data, v)
                    if pDot > currDot: return (pDot, p.data) # check if the nearest is current node found or the predecessor
                    else: return (currDot, curr.data)
                else:
                    curr = curr.left
            else:
                if curr.right is None:
                    s = self.successor(curr)
                    sDot = dot(s.data, v)
                    currDot = dot(curr.data, v)
                    if sDot > currDot: return (sDot, s.data) 
                    else: return (currDot, curr.data)
                else:
                    curr = curr.right

    def PrintTree(self):
        self.root.PrintFrom()

def AbsArc(v):  # v should be normalized
    #get a pseudo dot product for comparations that goes from 0(0°) to 1(360°)
    d = v[0] # cosine of v's angle with [1,0]T vector
    if v[1] >= 0:
        return (-d + 1)/4
    else:
        return (d + 3)/4

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def normalize(v):
    vLen = np.sqrt(v[0]**2+v[1]**2)
    if vLen is 0: return (0,0)
    return (v[0]/vLen, v[1]/vLen)

def softPath(graph):    #graph nodes containing vertex indexes tuples
    minNeededEdgePairDot = None
    # for each node get its data and that of all neighbours
    for nData, adjNodes in graph.adj.items():
        aTree = AngleSearchTree()
        nodeDirections = []
        nodeOpDirections = []
        for adjData, _ in adjNodes.items():
           dir = np.subtract(adjData, nData)  #direction from this node to this neighbour
           dir = normalize(dir)
           oppositeDir = (-dir[0], -dir[1])   #direction i want a node that pairs with me to be
           aTree.insert(dir)
           nodeDirections.append(dir)
           nodeOpDirections.append(oppositeDir)

        # now i have all directions of this node inserted, for each adj node I find 
        # the best other adj node (with angle most flat)
        for opDirection in nodeOpDirections:
            bestNode = aTree.findClosest(opDirection)
            bestDot = bestNode[0]
            if minNeededEdgePairDot is None or minNeededEdgePairDot > bestDot:
                minNeededEdgePairDot = bestDot

    return minNeededEdgePairDot

def tests():
    #tree1 = AngleSearchTree()
    #tree1.insert(normalize((1,1)))
    #tree1.insert(normalize((0,1)))
    #tree1.insert(normalize((-1,1)))
    #tree1.insert(normalize((-2,1)))
    #tree1.insert(normalize((-1,-1)))
    #tree1.insert(normalize((0,-1)))
    #tree1.insert(normalize((1,-1)))
    ##tree.PrintTree()
    #print(tree1.findClosest(normalize((0.1,1))))

    #tests with graph
    G = nx.Graph()
    G.add_edges_from([((0,0),(0,1)),((0,1),(1,1)),((1,1),(1,0)),((1,0),(0,0))])
    #for nData, adjNodes in G.adj.items():
    #    print("nodeData:", nData, "adjData:")
    #    for adjData, _ in adjNodes.items():
    #        print(adjData)
    print(softPath(G))

tests()