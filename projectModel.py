#https://python-sounddevice.readthedocs.io/en/latest/examples.html#play-a-sine-signal
import sys
import struct
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
import simpleaudio as sa
import math
import networkx as nx
from OBJloader import *
from graphLoader import *
from postman_problems.tests.utils import create_mock_csv_from_dataframe
from postman_problems.stats import calculate_postman_solution_stats
from postman_problems.solver import rpp, cpp
from softPath import *

# get an eulerian cicle from graph
def eulerCicle(graph):
    edgelist = nx.to_pandas_edgelist(graph, source='_node1', target='_node2') 
    edgelist_file = create_mock_csv_from_dataframe(edgelist)
    circuit_cpp_req, _ = cpp(edgelist_file)
    print('Print the CPP solution:')
    
    # solve chinese postman problem on graph
    # result circuit_cpp_req is in the form ('6.0', '2.0', 0, {'distance': 1.05144, 'id': 6, 'augmented': True})
    #                                   node1_|       |_node2 index 
    # keep only the nodes
    circuit_edges = []
    for e in circuit_cpp_req:
        circuit_edges.append((int(float(e[0])), int(float(e[1]))))

    return circuit_edges

def optimizedEulerian(model, circuit_edges):
    # optimize eulerian circuit using softPath optimizer
    circuit = []    #get nodes in circuit
    for n in circuit_edges:
        circuit.append(n[0])
    
    # get positions
    positions = []
    for n in model.vertices:
        positions.append((n[0], n[1]))

    optCircuit = softPath(model.graph, positions, circuit)

    # transform nodes circuit back in circuit edges
    circuit_edges = []
    for i in range(len(optCircuit)):
        circuit_edges.append((optCircuit[i], optCircuit[(i+1)%len(optCircuit)]))

    return circuit_edges

# get a list of all model's segments in draw order
def getSegs(model):
    # I expect a model with a graph, vertices list (xy positions)
    # I expect a nx graph with int node id and for each arc its length as attribute

    # solve chinese postman problem on graph
    # result is in the form (6, 2)
    #                  node1_| |_node2 index 
    circuit_edges = eulerCicle(model.graph)

    optimized_edges = optimizedEulerian(model, circuit_edges)

    # get vertex positions in order
    orderedVsegs = []
    for e in optimized_edges:
        orderedVsegs.append((model.vertices[e[0]], model.vertices[e[1]]))

    return orderedVsegs

def drawSegs(segments, drawTime, draws, amplitude, drawGraph):  #segs, time for a single draw, number of draws, scale(0..1)
    # calculate draw total length
    totalLength = 0
    for ends in segments:
        start = ends[0][0:2]
        end = ends[1][0:2]
        length = np.linalg.norm(np.subtract(end, start), 2)
        totalLength = totalLength + length
    print("total drawing length: " + str(totalLength))

    samplingRate = 44100
    drawing = [[],[]]    # single drawing
    drawingGraph = [[],[]]    # for showing in the graph
    signal = [[],[]]     # entire signal

    for ends in segments:
        # get ends projecting the third dimension isometrically
        start = ends[0][0:2]
        end = ends[1][0:2]
        length = np.linalg.norm(np.subtract(end, start), 2)
        # segment playtime is a fraction of the total drawTime
        segTime = (length / totalLength) * drawTime
        segSamples = segTime * samplingRate
        segSamples = round(segSamples, 0)   # number of samples must be an integer
        
        # fill drawing
        def smoothingFun(t):
            return (math.sin(math.pi * (t-0.5)) + 1) / 2 # sine accounts for actuator acceleration
            #return 1.0

        def positionInterp(startCoords, endCoords, completion):   # smooth linearly based on position
            return [startCoords[0] + (endCoords[0] - startCoords[0]) * smoothingFun(completion), startCoords[1] + (endCoords[1] - startCoords[1]) * smoothingFun(completion)]

        def velocityInterp(startCoords, endCoords, segTime, completion):   # get velocity to move from start to end (derivative of position increment) 
            #if completion > 0.5: return np.array([0.0,0.0])     # give time to decelerate at the end of the segment

            dx = (endCoords[0] - startCoords[0])
            dy = (endCoords[1] - startCoords[1])
            maxChange = np.max(np.abs([dx, dy]))
            if maxChange == 0.0: return np.array([0.0,0.0])
            return np.array([dx, dy]) * smoothingFun(completion*2) / maxChange

        if segSamples > 0:
            for i in range(int(segSamples)):
                segCompletion = i/segSamples
                #newVal = positionInterp(start, end, segCompletion)
                newVal = velocityInterp(start, end, segTime, segCompletion)
                drawing[0].append(newVal[0])
                drawing[1].append(newVal[1])
                # for graph drawing
                newValPosition = positionInterp(start, end, segCompletion)
                drawingGraph[0].append(newValPosition[0])
                drawingGraph[1].append(newValPosition[1])

    # duplicate drawing to get complete signal
    signal[0] = drawing[0] * draws
    signal[1] = drawing[1] * draws
    
    # write to wav file
    npArr = np.array(signal)

    #plot result
    if drawGraph:
        #plt.subplot(311)
        #plt.plot(drawingGraph[0], drawingGraph[1], "bo", signal[0], signal[1], "k")
        #plt.plot(drawingGraph[0], drawingGraph[1], "bo")
        graphWithColors(plt.figure().add_subplot(311), drawingGraph[0], drawingGraph[1])
        plt.subplot(312)
        plt.plot(np.linspace(0.0, drawTime, len(drawing[0])), drawing[0])
        plt.subplot(313)
        plt.plot(np.linspace(0.0, drawTime, len(drawing[1])), drawing[1])
        plt.show()
    else:
        # play audio
        npArr = np.ascontiguousarray(npArr.T, dtype = np.float32)
        audio = npArr * 32767 * amplitude / np.max(np.abs(npArr))
        audio = audio.astype(np.int16)
        play_obj = sa.play_buffer(audio, 2, 2, samplingRate)
        play_obj.wait_done()

def graphWithColors(subPlt, xArr, yArr):
    MAP='winter'
    NPOINTS = len(xArr)
    #defining an array of colors  
    colors = np.linspace(0, 255, NPOINTS)
    sizes = np.full(NPOINTS, 1.0)
    #applies the custom color map along with the color sequence
    subPlt.scatter(xArr, yArr, s=sizes, alpha=0.70, c= colors, cmap=plt.get_cmap('rainbow'))

# MAIN
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", action="store")
    parser.add_argument("-t", "--drawTime", action="store", type=float, default=0.1)
    parser.add_argument("-n", "--draws", action="store", type=int, default=200)
    parser.add_argument("-a", "--amplitude", action="store", type=float, default=1.0)
    parser.add_argument("--drawGraph", action="store_true", help="visualize graph")
    parser.add_argument("--fromGraphml", action="store_true", help="get graph from a graphml file instead")
    args = parser.parse_args()

    if args.fromGraphml == True:
        model = graphml(args.file)
    else:
        model = OBJ(args.file)
    segs = getSegs(model)

    drawSegs(segs, args.drawTime, args.draws, args.amplitude, args.drawGraph)

if __name__ == "__main__":
    main()