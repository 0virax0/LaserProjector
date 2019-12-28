# Input: eulerian networkX graph 
# Output: an eulerian path with angles relaxed

# Maximizing the minimum angle in the path is not a simple problem, instead I 
# relax angles formed between arcs crossing a single node:
# I can safely switch loops walking in reverse order starting and ending in
# the same node because the complete loop will still be an eulerian cicle

import numpy as np
import math
import networkx as nx

#class Step:
   #def __init__(self, nIndex = None, prevS = None, nextS = None, neighbourDir = None, neighbourArc = None):
        #self.nIndex = nIndex # index of the node
        #self.neighbourDir = neighbourDir # here I store temporarily my position relative to the node I am analyzing
        #self.neighbourArc = neighbourArc # temporary arc formed with analyzing node
        #self.prevS = prevS    
        #self.nextS = nextS


class Step:
   def __init__(self, nIndex = None, neighbourDir = None, neighbourArc = None):
        self.nIndex = nIndex # index of the node
        self.neighbourDir = neighbourDir # here I store temporarily my position relative to the node I am analyzing
        self.neighbourArc = neighbourArc # temporary arc formed with analyzing node

class Path:
    def __init__(self, step = None, length = 0):
        self.head = step
        self.length = length
    
    def insertBetween(self, newStep, step1, step2):
        if step1 is None and step2 is None:
            return

        newStep.prevS = step1
        if step1 is not None:
            step1.nextS = newStep
        newStep.nextS = step2
        if step2 is not None:
            step2.nextS = newStep
        
    def estract(self, step):
        if step.prevS is not None:
            step.prevS.nextS = step.nextS
        if step.nextS is not None:
            step.nextS.prevS = step.prevS
        
        step.prevS = None
        step.nextS = None

    def print(self):
        node = self.head
        while (node is not None):
          print(node.data),
          node = node.next

def softPath(graph, nodePositions, initPath):    #graph nodes containing vertex indexes and list of node positions ordered by index in the graph
    # construct Path double linked list from initPath
    if len(initPath) <= 1:
        return
    
    #path = Path()
    #preparedNode = None 
    #for index in initPath:
        #newNode = Step(index, preparedNode, None, None, None)
        #if preparedNode is not None:
            #preparedNode.nextS = newNode
        #newNode.prevS = preparedNode
        #preparedNode = newNode
        ##first node
        #if path.head is None:
            #path.head = newNode
    ## closing the list in a loop
    #preparedNode.nextS = path.head
    #path.head.prevS = preparedNode
    #path.length = len(initPath) # set list total length

    path = []
    for nIndex in initPath:
        path.append(Step(nIndex, None, None))
    pathLen = len(path)

    # for each node get its data and that of all neighbours
    for nIndex, _ in graph.adj.items():
        # aggregate nodes positions in the path (may have scrumbled from previous step and I need them in order)
        aggrNindex = aggregateNodes(path) # nIndex(index of the node in graph) : i (index of step in the path)

        # get every neighbour's step so I can update data directly in the list
        neighbours = getNeighbours(aggrNindex, path, nIndex, pathLen)
        if len(neighbours) <= 2:
            continue    # for nodes with 2 vertices I do not need scrambling
        if len(neighbours)%2 == 1:
            raise Exception("every node should have an even number of arcs for the graph to be eulerian")

        setNeighboursInfo(nIndex, path, neighbours, nodePositions)

        # I create a map to quickly find the other end of the loop (needs neighbours to be ordered sequentially with respect to the path)
        loopsMap = getLoops(path, neighbours)

        # I order neighbours based on their arc relative to this node
        sortedNeighbours = orderNeighbours(path, neighbours)

        # index of sortedNeghbours: I divide neighbours in 2 so that half are in half circle
        divisionIndexes = divideNeighbours(path, sortedNeighbours)
        #print(divisionIndexes)

        # I pair neighbours so each has the other the most far possible in the circle (skipping those already connected via a loop) 
        # this way when i cross this node i do with maximal angles
        pairedNeighbours = pairNeighbours(path, sortedNeighbours, divisionIndexes, loopsMap)
        print(pairedNeighbours)

def aggregateNodes(path):
    # from path i organize nIndexes aggregating same nIndex so that i recognize
    # every time I pass on that node
    aggrNindex = {} # nIndex: (index of the node in graph)
    #currStep = path.head
    #for _ in range(path.length):
        #if currStep.nIndex not in aggrNindex:
            #aggrNindex[currStep.nIndex] = [currStep] 
        #else:
            #aggrNindex[currStep.nIndex].append(currStep)

        #currStep = currStep.nextS
    for i, step in enumerate(path):
        if step.nIndex not in aggrNindex:
            aggrNindex[step.nIndex] = [i] 
        else:
            aggrNindex[step.nIndex].append(i)
    
    return aggrNindex

def getNeighbours(aggrNindexTable, path, nIndex, pathLen):
    neighbours = [] # pointers to their steps
    #for reencouter in aggrNindexTable[nIndex]:
        #if reencouter.prevS is not None:
            #neighbours.append(reencouter.prevS)
        #if reencouter.nextS is not None:
            #neighbours.append(reencouter.nextS)
    for reencouter in aggrNindexTable[nIndex]:
        neighbours.append((reencouter-1) % pathLen)
        neighbours.append((reencouter+1) % pathLen)
    
    return neighbours

def setNeighboursInfo(nIndex, path, neighbours, nodePositions):
    for neighbour in neighbours:
        dir = np.subtract(nodePositions[path[neighbour].nIndex], nodePositions[nIndex])  #direction from this node to this neighbour
        dir = normalize(dir)
        path[neighbour].neighbourDir = dir
        # calculate absolute dot of neighbour direction
        absDot = arcBetween((1,0), dir)
        path[neighbour].neighbourArc = absDot

def getLoops(path, neighbours):
    # I suppose the neighbours are in order with respect to the current path
    loopsMap = {}

    for i in range(int((len(neighbours))/2)):
        # ends reside one next to the other starting from the second element
        ends = (neighbours[(i*2+1) % len(neighbours)], neighbours[(i*2+2) % len(neighbours)])
        loopsMap[ends[0]] = ends[1]
        loopsMap[ends[1]] = ends[0]

    return loopsMap

def orderNeighbours(path, neighbours):
    return sorted(neighbours, key=lambda stepIndex: path[stepIndex].neighbourArc)

def divideNeighbours(path, sortedNeighbours):
    length = len(sortedNeighbours)
    startIndex = 0
    def getDir(index):
        return path[sortedNeighbours[index]].neighbourDir

    while arcBetween(getDir(startIndex), getDir((startIndex + math.ceil((length/2) - 1)) % length)) >= 0.5 or arcBetween(getDir(startIndex), getDir((startIndex + math.floor((length/2))) % length)) < 0.5:
        startIndex = (startIndex + 1) % length
        #print("arc before ",arcBetween(getDir(startIndex), getDir((startIndex + math.ceil((length/2) - 1)) % length)), " arc after ",arcBetween(getDir(startIndex), getDir((startIndex + math.floor((length/2))) % length)))
    return (startIndex, (startIndex + math.ceil((length/2) - 1)) % length)

def pairNeighbours(path, sortedNeighbours, divisionIndexes, loopsMap):
    topHalf = []
    bottomHalf = []
    fullLen = len(sortedNeighbours)
    halfLen = math.floor(len(sortedNeighbours)/2)

    for i in range(halfLen):
        topHalf.append(sortedNeighbours[(divisionIndexes[0]+i) % fullLen])
        # bottomHalf has to be reversed (so is in the circle)
        bottomHalf.append(sortedNeighbours[(divisionIndexes[1]+i+1) % fullLen]) 

    # merge halfs getting couples (not forming a loop)
    neighCouples = []
    indexBottom = 0
    while len(topHalf) > 0:
        if loopsMap[topHalf[0]] == bottomHalf[indexBottom]:
            indexBottom = indexBottom + 1 #they form a loop, skip
        else:
            neighCouples.append((topHalf.pop(0), bottomHalf.pop(indexBottom)))
            indexBottom = 0
    return neighCouples

def arcBetween(v1, v2): # v1, v2 should be normalized
    #get a pseudo dot product for comparations that goes from 0(0°) to 1(360°)
    dot = np.dot(v1,v2)

    # search if v2 is under v1
    v1Perp = np.array([-v1[1], v1[0]])
    perpDot = np.dot(v1Perp, v2) # measures the sine, if positive v2 is above with v1
    if perpDot > 0:
        return (-dot + 1)/4
    else:
        return (dot + 3)/4

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def tests():
    G = nx.Graph() # indexes graph
    #G.add_edges_from([(0,1),(1,2),(2,3),(3,0)]) # square based on indexes
    #positions = [(0,0),(0,1),(1,1),(1,0)]    # positions of nodes
    #initPath = [0,1,2,3] # not closed path

    G.add_edges_from([(0,1),(1,2),(3,1),(1,4),(0,2),(3,4),(2,3),(4,0),(2,5),(5,3),(4,6),(6,0)]) # square with center
    positions = [(0,0),(0.5,0.5),(0,1),(1,1),(1,0),(0.5,2),(0.5,-1)]    # positions of nodes
    initPath = [0,1,2,3,1,4,0,6,4,3,5,2] # not closed path

    #G.add_edges_from([(0,1),(1,2),(2,0),(0,3),(3,4),(4,0),(0,5),(5,6),(6,0)]) # single node with many arcs
    #positions = [(0,0),(1,-0.9),(-0.5,-1),(-1,0.3),(-1,0.8),(-0.3,1),(0.4,1)]    # positions of nodes
    #initPath = [0,1,2,0,3,4,0,5,6] # not closed path

    softPath(G, positions, initPath)

tests()