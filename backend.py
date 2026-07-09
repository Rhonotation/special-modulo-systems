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

    def get_max_y(self):
        return np.max(np.array([item[1] for item in list(self.cells.values)]))
    
    def neighbors(self):
        celllocs = np.array(list(self.cells.values))
        neighborlist = []
        for cellloc in celllocs:
            neighborlist.extend([[cellloc[0] + 1, cellloc[1]],
                                 [cellloc[0] - 1, cellloc[1]],
                                 [cellloc[0], cellloc[1] + 1],
                                 [cellloc[0], cellloc[1] - 1]])
        # Expansion Complete
        maxx = self.get_max_x()
        maxy = self.get_max_y()
        neighborlist = np.array(neighborlist)
        neighborlist = neighborlist[((0 <= neighborlist[0]) and (neighborlist[0] <= maxx)) and((0 <= neighborlist[0]) and (neighborlist[0] <= maxy))]
        # Bounds Filtering Complete
        neighborlist = neighborlist[neighborlist not in celllocs]
        # Overlap Filtering Complete
        neighborlist = np.unique(neighborlist)