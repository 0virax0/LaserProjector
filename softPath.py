# Input: eulerian networkX graph 
# Output: an eulerian path with angles relaxed

# Maximizing the minimum angle in the path is not a simple problem, instead I 
# relax angles formed between arcs crossing a single node:
# I can safely switch loops walking in reverse order starting and ending in
# the same node because the complete loop will still be an eulerian cicle

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
        if pointers[1] in self.dictionary:
            self.dictionary[pointers[1]][0] = pointers[0]
    
    def swallowCopy(self):
        copy = AssociativeList()
        copy.dictionary = self.dictionary.copy()
        copy.first = self.first
        copy.last = self.last
        return copy
    
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
        #print(divisionIndexes)

        # I pair neighbours so each has the other the most far possible in the circle (skipping those already connected via a loop) 
        # this way when i cross this node i do with maximal angles
        # returns a map between paired neighbours
        pairedNeighbours = pairNeighbours(path, sortedNeighbours, divisionIndexes, loopsMap)

        # recreate path reversing or translating loops
        #path = rebuildPath(path, loopsMap, pairedNeighbours, nIndex)
        print("--------endnode------------")

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
    def getDir(index):
        return path[sortedNeighbours[index]].neighbourDir

    while arcBetween(getDir(startIndex), getDir((startIndex + math.ceil((length/2) - 1)) % length)) >= 0.5 or arcBetween(getDir(startIndex), getDir((startIndex + math.floor((length/2))) % length)) < 0.5:
        startIndex = (startIndex + 1) % length
        #print("arc before ",arcBetween(getDir(startIndex), getDir((startIndex + math.ceil((length/2) - 1)) % length)), " arc after ",arcBetween(getDir(startIndex), getDir((startIndex + math.floor((length/2))) % length)))
    return (startIndex, (startIndex + math.ceil((length/2) - 1)) % length)

def pairNeighbours(path, sortedNeighbours, divisionIndexes, loopsMap):
    neighCouples = {}
    topHalf = AssociativeList()
    bottomHalf = AssociativeList()
    fullLen = len(sortedNeighbours)
    halfLen = math.floor(len(sortedNeighbours)/2)

    for i in range(halfLen):
        topHalf.append(sortedNeighbours[(divisionIndexes[0]+i) % fullLen])
        # bottomHalf has to be reversed (so is in the circle)
        bottomHalf.append(sortedNeighbours[(divisionIndexes[1]+i+1) % fullLen]) 

    # Connect nodes in the top half with those on the bottom half so that including already present semi-loops
    # I form a complete tour of the nodes without short circuiting. Doing so I try to maximize angles formed
    # by keeping nodes with the same index (top and bottom) nearby.
    # To do so I use an euristic: set first and last node first as they can form the narrowest angles with 
    # other nodes, then the others. To not form a short circuit I create incomplete paths that will be merged
    # on every step in a divide and conquer manner 

    # at every step keep only not already connected nodes on top and bottom lists
    # at every step copy current lists so I can remove also the ends of the loops (for not short circuiting)

    while topHalf.len() > 1 and bottomHalf.len() > 1 :
        stepTop = topHalf.swallowCopy()
        stepBottom = bottomHalf.swallowCopy()

       # connect extreme nodes excluding ends of loops until none remains  
        while stepTop.len() > 0 :
            # first

            firstTop = stepTop.first
            # connection
            firstBottom = stepBottom.first 
            firstTPrev = loopsMap[firstTop]
            firstBSucc = loopsMap[firstBottom]

            # remove nodes so I cannot reach them again (short circuiting) 
            stepTop.delete(firstTop)
            stepBottom.delete(firstBottom)
            stepTop.delete(firstTPrev)
            stepBottom.delete(firstTPrev) # I don't know where to find it
            stepTop.delete(firstBSucc)
            stepBottom.delete(firstBSucc) 
            # remove also from top and bottom halfs so in the successive iterations I don't consider them again
            topHalf.delete(firstTop)
            bottomHalf.delete(firstBottom)
            # add to couples list
            neighCouples[firstTop] = firstBottom
            neighCouples[firstBottom] = firstTop

            # last

            lastTop = stepTop.last
            # connection
            lastBottom = stepBottom.last 
            lastTPrev = loopsMap[lastTop]
            lastBSucc = loopsMap[lastBottom]

            # remove nodes so I cannot reach them again (short circuiting) 
            stepTop.delete(lastTop)
            stepBottom.delete(lastBottom)
            stepTop.delete(lastTPrev)
            stepBottom.delete(lastTPrev) # I don't know where to find it
            stepTop.delete(lastBSucc)
            stepBottom.delete(lastBSucc) 
            # remove also from top and bottom halfs so in the successive iterations I don't consider them again
            topHalf.delete(lastTop)
            bottomHalf.delete(lastBottom)
            # add to couples list
            neighCouples[lastTop] = lastBottom
            neighCouples[lastBottom] = lastTop

    # add last couple
    neighCouples[topHalf.first] = bottomHalf.first
    neighCouples[bottomHalf.first] = topHalf.first

    # merge halfs getting couples (not forming a loop)
    #indexBottom = 0
    #topDone = {}    # already taken neighbours from the top
    #neighCouples = {}
    #while len(topHalf) > 0:
        ## if current top element forms a loop with bottom or bottom forms a path with any other top element already kept
        ## skip as I cannot have a crossing inside a loop and if I have paths with top elements I short circuit (exept for last element)
        #currTop = topHalf[0]
        #currBottom = bottomHalf[indexBottom]
        #print("indexBottom: ",indexBottom, "top: ", topHalf, "bottom: ", bottomHalf ,"topDone:",topDone,"bottomOtherEnd:",loopsMap[(currBottom, False)][0])
        #if (loopsMap[(currTop, False)][0] == currBottom) or ((loopsMap[(currBottom, False)][0] in topDone or loopsMap[(currBottom, True)][0] in topDone) and len(topHalf) > 1):
            #indexBottom = indexBottom + 1 #they form a loop, skip
        #else:
            #n1 = topHalf.pop(0)
            #n2 = bottomHalf.pop(indexBottom)
            #topDone[n1] = True # save to avoid connecting a bottom node to a top done, short circuiting
            #n1d = (n1, False)
            #n2d = (n2, False)
            ## if I already found this neighbour I am returning from the same arc(non euclidean path), I need to 
            ## differenciate from the other so that later merge will work
            #if n1d in neighCouples:
                #n1d = (n1, True)

            #if n2d in neighCouples:
                #n2d = (n2, True)

            #neighCouples[n1d] = n2d
            #neighCouples[n2d] = n1d
            #print((path[n1d[0]].nIndex,n1d[1]),",",(path[n2d[0]].nIndex,n2d[1]))
            #indexBottom = 0
    return neighCouples

def rebuildPath(path, loopsMap, pairedNeighbours, nIndex):
    # i read steps following pairedNeighbours and add nIndex between them
    print("prima: ",pathToIndexes(path))
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
        print(nextN)
        newPath.append(Step(nIndex, None, None))
        readThenWriteFrom(nextN[0], loopsMap[nextN][0])
        nextN = pairedNeighbours[loopsMap[nextN]]
    
    print("dopo:  ",pathToIndexes(newPath))
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

    #G.add_edges_from([(0,1),(1,2),(3,1),(1,4),(0,2),(3,4),(2,3),(4,0),(2,5),(5,3),(4,6),(6,0)]) # square with center
    #positions = [(0,0),(0.5,0.5),(0,1),(1,1),(1,0),(0.5,2),(0.5,-1)]    # positions of nodes
    #initPath = [0,1,2,3,1,4,0,6,4,3,5,2] # not closed path

    #G.add_edges_from([(0,1),(1,2),(2,0),(0,3),(3,4),(4,0),(0,5),(5,6),(6,0)]) # single node with many arcs
    #positions = [(0,0),(1,-0.9),(-0.5,-1),(-1,0.3),(-1,0.8),(-0.3,1),(0.4,1)]    # positions of nodes
    #initPath = [0,1,2,0,3,4,0,5,6] # not closed path

    G.add_edges_from([(0,1),(1,2),(2,0),(0,3),(0,4),(0,5)]) # non eulerian cicle
    positions = [(0,0),(0,1),(0.7,0.3),(0.5,-0.6),(-0.5,-0.6),(-0.7,0.3)]   
    initPath = [0,1,2,0,3,0,4,0,5] 
    softPath(G, positions, initPath)

tests()