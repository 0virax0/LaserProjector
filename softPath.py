# Input: eulerian networkX graph 
# Output: an eulerian path with angles relaxed

# Maximizing the minimum angle in the path is not a simple problem, instead I 
# relax angles formed between arcs crossing a single node:
# I can safely switch loops walking in reverse order starting and ending in
# the same node because the complete loop will still be an eulerian cicle

import copy
import numpy as np
import math
import networkx as nx

class Step:
    def __init__(self, nIndex = None, neighbourDir = None, neighbourArc = None):
        self.nIndex = nIndex # index of the node
        self.neighbourDir = neighbourDir # here I store temporarily my position relative to the node I am analyzing
        self.neighbourArc = neighbourArc # temporary arc formed with analyzing node

class AssociativeList:
    # associates obj => (prevObj, succObj)
    dictionary = {}
    first = None
    last = None

    def __init__(self):
        self.dictionary = {}
        self.first = None
        self.last = None

    def ins(self, obj, prev, succ):
        self.dictionary[obj] = [prev, succ]
    
    def append(self, obj):
        if self.first is None:
            self.ins(obj, None, None)
            self.first = obj
            self.last = obj
        else:
            self.dictionary[self.last][1] = obj
            self.ins(obj, self.last, None)
            self.last = obj

    
    def delete(self, obj):
        if obj not in self.dictionary:
            return

        pointers = self.dictionary[obj]
        del self.dictionary[obj]
        if pointers[0] in self.dictionary:
            self.dictionary[pointers[0]][1] = pointers[1]
        else: # I'm the first
            self.first = pointers[1]

        if pointers[1] in self.dictionary:
            self.dictionary[pointers[1]][0] = pointers[0]
        else: # I'm the last
            self.last = pointers[0]
    
    def len(self):
        return len(self.dictionary)

def softPath(graph, nodePositions, initPath):    #graph nodes containing vertex indexes and list of node positions ordered by index in the graph
    # construct Path double linked list from initPath
    if len(initPath) <= 1:
        return

    path = []
    for nIndex in initPath:
        path.append(Step(nIndex, None, None))
    pathLen = len(path)

    # for each node get its data and that of all neighbours
    for nIndex, _ in graph.adj.items():
        print("nIndex:", nIndex)
        # aggregate nodes positions in the path (may have scrumbled from previous step and I need them in order)
        aggrNindex = aggregateNode(path, nIndex) # find every occurrence of the node in path and return the index

        # get every neighbour's step so I can update data directly in the list
        neighbours = getNeighbours(aggrNindex, path, pathLen)
        if len(neighbours) <= 2:
            continue    # for nodes with 2 arcs I do not need scrambling

        setNeighboursInfo(nIndex, path, neighbours, nodePositions)

        # I create a map to quickly find the other end of the loop (needs neighbours to be ordered sequentially with respect to the path)
        loopsMap = getLoops(path, neighbours)

        # I order neighbours based on their arc relative to this node
        sortedNeighbours = orderNeighbours(path, neighbours)

        # index of sortedNeghbours: I divide neighbours in 2 so that half are in half circle
        divisionIndexes = divideNeighbours(path, sortedNeighbours)

        # I pair neighbours so each has the other the most far possible in the circle (skipping those already connected via a loop) 
        # this way when i cross this node i do with maximal angles
        # returns a map between paired neighbours
        pairedNeighbours = pairNeighbours(path, sortedNeighbours, divisionIndexes, loopsMap)

        # recreate path reversing or translating loops
        path = rebuildPath(path, loopsMap, pairedNeighbours, nIndex)
        print("--------endnode------------")
    
    return pathToIndexes(path)

def aggregateNode(path, nIndex):
    # from path I organize nIndexes aggregating same nIndex so that I recognize
    # every time I pass on that node
    aggrNindex = [] 
    for i, step in enumerate(path):
        if step.nIndex == nIndex:
            aggrNindex.append(i)
    
    return aggrNindex

def getNeighbours(aggrNindex, path, pathLen):
    neighbours = [] # pointers to their steps
    for reencouter in aggrNindex:
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
        if ends[0] == ends[1]: # distinguish duplicates so it works even if cicle is not eulerian
                            # as both ands will be the same
            loopsMap[(ends[0], False)] = (ends[0], True)
            loopsMap[(ends[0], True)] = (ends[0], False)
        else:
            loopsMap[(ends[0], False)] = (ends[1], False)
            loopsMap[(ends[1], False)] = (ends[0], False)

    return loopsMap

def orderNeighbours(path, neighbours):
    return sorted(neighbours, key=lambda stepIndex: path[stepIndex].neighbourArc)

def divideNeighbours(path, sortedNeighbours):
    length = len(sortedNeighbours)
    startIndex = 0

    def getInnerIndex(startIndex):
        return (startIndex + math.ceil((length/2) - 1)) % length
    def getOuterIndex(startIndex):
        return (startIndex + math.floor((length/2))) % length

    def getDir(index):
        return path[sortedNeighbours[index]].neighbourDir
    
    def arcBetweenIndexes(startIndex, endIndex):
        def rotateBias(dir, index):
            sin = 0.001 * index
            cos = (1-sin)
            r = (dir[0]*cos - dir[1]*sin, dir[0]*sin + dir[1]*cos)
            return r
        # add bias to avoid overlappings
        return arcBetween(rotateBias(getDir(startIndex), startIndex), rotateBias(getDir(endIndex), endIndex))
    
    def condition():
        innerArc = arcBetweenIndexes(startIndex, getInnerIndex(startIndex)) 
        outerArc = arcBetweenIndexes(startIndex, getOuterIndex(startIndex)) 
        return innerArc >= 0.5 or outerArc < 0.5 

    while condition():
        startIndex = (startIndex + 1) % length
        #print("arc before ",arcBetweenIndexes(startIndex, getInnerIndex(startIndex)), " arc after ",arcBetweenIndexes(startIndex, getOuterIndex(startIndex)) )
    return (startIndex, (startIndex + math.ceil((length/2) - 1)) % length)

def pairNeighbours(path, sortedNeighbours, divisionIndexes, loopsMap):
    neighCouples = {}
    currLoopsMap = copy.deepcopy(loopsMap)
    topHalf = AssociativeList()
    bottomHalf = AssociativeList()
    fullLen = len(sortedNeighbours)
    halfLen = math.floor(len(sortedNeighbours)/2)

    # copy divided indexes
    for i in range(halfLen):
        topElem = sortedNeighbours[(divisionIndexes[0]+i) % fullLen]
        # duplicates in case of non euclidean path, i need to distinguish them
        if (topElem, False) not in topHalf.dictionary:
            topHalf.append((topElem, False))
        else: 
            topHalf.append((topElem, True))
        # bottomHalf has to be reversed (so is in the circle)
        bottomElem = sortedNeighbours[(divisionIndexes[1]+i+1) % fullLen]
        if (bottomElem, False) not in bottomHalf.dictionary and (bottomElem, False) not in topHalf.dictionary:
            bottomHalf.append((bottomElem, False)) 
        else:
            bottomHalf.append((bottomElem, True)) 

    # Connect nodes in the top half with those on the bottom half so that including already present semi-loops
    # I form a complete tour of the nodes without short circuiting. Doing so I try to maximize angles formed
    # by keeping nodes with the same index (top and bottom) nearby.
    # To do so I use an euristic: set first and last node first as they can form the narrowest angles with 
    # other nodes, then the others. To not form a short circuit I create incomplete paths that will be merged
    # on every step in a divide and conquer manner 

    # at every step keep only not already connected nodes on top and bottom lists
    # at every step copy current lists so I can remove also the ends of the loops (for not short circuiting)

    while topHalf.len() >= 2 and bottomHalf.len() >= 2 :
        stepTop = copy.deepcopy(topHalf)
        stepBottom = copy.deepcopy(bottomHalf)

        firstext = True # true for first element, false for last

    # connect ext nodes excluding ends of loops until none remains  
        while stepTop.len() >= 1 and stepBottom.len() >= 1: #and (stepTop.len()+stepBottom.len())>=4:
            ### first
            # match and remove nodes so I cannot reach them again (short circuiting) 
            if firstext == True:
                extTop = stepTop.first
            else:
                extTop = stepTop.last
            if extTop is None:  # elements are exausted before time
                break   
            stepTop.delete(extTop)

            extTPrev = currLoopsMap[extTop]
            stepTop.delete(extTPrev)
            stepBottom.delete(extTPrev) # I don't know where to find it

            # connection
            if firstext == True:
                extBottom = stepBottom.first 
            else:
                extBottom = stepBottom.last 
            if extBottom is None:
                break   # elements are exausted before time
            stepBottom.delete(extBottom)

            extBSucc = currLoopsMap[extBottom]
            stepTop.delete(extBSucc)
            stepBottom.delete(extBSucc) # I don't know where to find it

            # change loops ends (extending them)
            del currLoopsMap[extTop]  # now connected, no longer need it
            del currLoopsMap[extBottom]
            currLoopsMap[extTPrev] = extBSucc
            currLoopsMap[extBSucc] = extTPrev

            # add to couples list
            neighCouples[extTop] = extBottom
            neighCouples[extBottom] = extTop

            # remove also from top and bottom halfs so in the successive iterations I don't consider them again
            topHalf.delete(extTop)
            bottomHalf.delete(extBottom)

            firstext = not firstext # switch between first and last element (that have more priority for being analyzed)

    # add last couple
    neighCouples[topHalf.first] = bottomHalf.first
    neighCouples[bottomHalf.first] = topHalf.first

    #print("neighCouples:", neighCouples)
    return neighCouples

def rebuildPath(path, loopsMap, pairedNeighbours, nIndex):
    # i read steps following pairedNeighbours and add nIndex between them
    #print("prima: ",pathToIndexes(path))
    newPath = []
    def readThenWriteFrom(start, end):
        if start != end:
            if path[(start-1) % len(path)].nIndex == nIndex: #loop is in the inside
                currIndex = start
                while currIndex != end:
                    newPath.append(path[currIndex])
                    currIndex = (currIndex + 1) % len(path)
            else: #loop on the outside
                currIndex = start
                while currIndex != end:
                    newPath.append(path[currIndex])
                    currIndex = (currIndex - 1) % len(path)

        newPath.append(path[end])

    nextN = next(iter(pairedNeighbours)) # get a starting neighbour
    for _ in range(math.floor(len(pairedNeighbours.keys())/2)):
        newPath.append(Step(nIndex, None, None))
        readThenWriteFrom(nextN[0], loopsMap[nextN][0])
        nextN = pairedNeighbours[loopsMap[nextN]]
    
    #print("dopo:  ",pathToIndexes(newPath))
    return newPath

def pathToIndexes(path):
    Gindexes = []
    for step in path:
        Gindexes.append(step.nIndex)
    return Gindexes

def arcBetween(v1, v2): # v1, v2 should be normalized
    #get a pseudo dot product for comparations that goes from 0(0°) to 1(360°)
    dot = np.dot(v1,v2)

    # search if v2 is under v1
    v1Perp = np.array([-v1[1], v1[0]])
    perpDot = np.dot(v1Perp, v2) # measures the sine, if positive v2 is above with v1
    if perpDot >= 0:
        return (-dot + 1)/4
    else:
        return (dot + 3)/4

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return [0,1]
    return v / norm

def tests():
    G = nx.Graph() # indexes graph
    #G.add_edges_from([(0,1),(1,2),(2,3),(3,0)]) # square based on indexes
    #positions = [(0,0),(0,1),(1,1),(1,0)]    # positions of nodes
    #initPath = [0,1,2,3] # not closed path

    #G.add_edges_from([(0,1),(1,2),(3,1),(1,4),(0,2),(3,4),(2,3),(4,0),(2,5),(5,3),(4,6),(6,0)]) # square with center
    #positions = [(0,0),(0.5,0.5),(0,1),(1,1),(1,0),(0.5,2),(0.5,-1)]    # positions of nodes
    #initPath = [0,1,2,3,1,4,0,6,4,3,5,2] # not closed path

    #G.add_edges_from([(0,1),(1,2),(2,0),(0,3),(3,4),(4,0),(0,5),(5,6),(6,0),(0,7),(7,8),(8,0)]) # single node with many arcs
    #positions = [(0,0),(1,-0.9),(-0.5,-1),(-1,0.3),(-1,0.8),(-0.3,1),(0.4,1),(1,0.3),(1,0)]    # positions of nodes
    #initPath = [0,1,2,0,3,4,0,5,6,0,7,8] # not closed path

    #G.add_edges_from([(0,1),(1,2),(2,0),(0,3),(0,4),(0,5)]) # non eulerian cicle
    #positions = [(0,0),(0,1),(0.7,0.3),(0.5,-0.6),(-0.5,-0.6),(-0.7,0.3)]   
    #initPath = [0,1,2,0,3,0,4,0,5] 

    G.add_edges_from([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3),(3,6),(6,7),(7,3)]) # another non eulerian cicle
    positions = [(0.5,2),(1.5,2),(1,1.5),(1,1),(0,0.5),(0.7,0.2),(1.3,0.2),(2,0.5)]   
    initPath = [0,1,2,3,4,5,3,6,7,3,2] 

    softPath(G, positions, initPath)