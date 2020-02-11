import numpy as np
import networkx as nx

class OBJ:
    """Loads an OBJ file without materials """
    vertices = []
    normals = []
    texcoords = []
    faces = []

    segments = []   #segments with vertex index
    vSegments = []  #segments in vertex coordinates  

    graph = {}

    def __init__(self, filename):

        # write verices, normals, texcoords, faces from model
        for line in open(filename, "r"):
            if line.startswith("#"): continue
            values = line.split()
            if not values: continue
            if values[0] == "v":
                self.vertices.append(list(map(float, values[1:4])))
            elif values[0] == "vn": 
                self.normals.append(list(map(float, values[1:4])))
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []   #in the format (face, norms, texcoords)
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))
        self.getSegs()
        self.getGraph()
    
    # get segments and vSegments
    def getSegs(self):
        encounteredEdge = dict()   #we don't want a graph with repeated edges

        for face in self.faces:
            fList = face[0]  #list with vertex indexes as they appear unfolding the face
            fList.append(face[0][0])    #repeat first vertex to close the face

            for i in range(len(fList)-1):
                start = fList[i]
                end = fList[i+1]
                # register encounter
                if ((start,end) not in encounteredEdge) and ((end,start) not in encounteredEdge):
                    encounteredEdge[(start,end)] = True
                    self.segments.append((start, end))

        # fill segments indexes with vertex data
        for ends in self.segments:
            self.vSegments.append((self.vertices[ends[0]-1], self.vertices[ends[1]-1]))

    def getGraph(self):
        # use segments to create an edge graph and calculate distances with vSegments 
        self.graph = nx.MultiGraph()
        for i in range(len(self.segments)):
            length = np.linalg.norm(np.subtract(self.vSegments[i][1], self.vSegments[i][0]), 2)
            self.graph.add_edge(*self.segments[i], distance = length) # graph node name is the (int) index in segments

