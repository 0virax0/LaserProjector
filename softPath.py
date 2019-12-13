# Input: eulerian networkX graph 
# Output: an eulerian path with angles relaxed

# Maximizing the minimum angle in the path is not a simple problem, instead I 
# relax angles formed between arcs crossing a single node:
# I can safely switch loops walking in reverse order starting and ending in
# the same node because the complete loop will still be an eulerian cicle

import numpy as np
import math
import networkx as nx

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
    v1 = normalize(np.array([1, 1]))
    v2 = normalize(np.array([1, -1]))
    print(arcBetween(v1,v2))

tests()