class OBJ:
    def __init__(self, filename):
        """Loads an OBJ file without materials """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

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

