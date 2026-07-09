import numpy as np

class Piece():
    def __init__(self, seq=None):
        self.cells = {}
        if seq == None:
            self.len = 0
        else:
            self.len = len(seq)
    
    def get_max_x(self):
        return np.max(np.array([item[0] for item in list(self.cells.values)]))
    
    def neighbors(self):
        celllocs = list(self.cells.values)
        neighborlist = []
        for cellloc in celllocs:
            neighborlist.extend([[cellloc[0] + 1, cellloc[1]],
                                 [cellloc[0] - 1, cellloc[1]],
                                 [cellloc[0], cellloc[1] + 1],
                                 [cellloc[0], cellloc[1] - 1]])