import numpy as np
from functools import total_ordering
from collections.abc import Iterable

@total_ordering
class Piece():
    def __init__(self, seq=None):
        '''Initializes self.'''
        self.cells = {0:[0,0]}
        self.len = 1
        self.seq = []
        if seq:
            self.growseq(seq)

    def __str__(self):
        '''Turns self into a string.'''
        stringlst = np.array([[" " for _ in range(self.get_max_x() + 1)] for _ in range(self.get_max_y() + 1)])
        for num, cell in self.cells.items():
            stringlst[cell[1], cell[0]] = str(num) # we index y before x
        string = np.apply_along_axis(lambda x: "\t".join(x), 1, stringlst)
        return "\n".join(string)
    
    def get_max_x(self):
        '''Returns max x in self.'''
        return np.max(np.array([item[0] for item in list(self.cells.values())]))

    def get_max_y(self):
        '''Returns max y in self.'''
        return np.max(np.array([item[1] for item in list(self.cells.values())]))
    
    def get_cells(self):
        '''Returns cells in self.'''
        return self.cells

    def get_len(self):
        return self.len
    
    def neighbors(self):
        '''Gets neighbors of self.'''
        celllocs = np.array(list(self.cells.values()))
        neighborlist = []
        for cellloc in celllocs:
            neighborlist.extend([[cellloc[0] + 1, cellloc[1]],
                                 [cellloc[0] - 1, cellloc[1]],
                                 [cellloc[0], cellloc[1] + 1],
                                 [cellloc[0], cellloc[1] - 1]])
        # Expansion Complete
        current_cells_set = set(tuple(c) for c in celllocs)
        neighborlist = np.array([n for n in neighborlist if tuple(n) not in current_cells_set])
        # Overlap Filtering Complete
        neighborlist = np.unique(neighborlist, axis=0)
        # Uniqueness Filtering Complete
        neighborlist = neighborlist[np.lexsort((neighborlist[:, 0], neighborlist[:, 1]))]
        # Sorting Complete
        return neighborlist.tolist()
    
    def grow(self, idx):
        '''Adds neighbor idx to self.'''
        celllocs = np.array(list(self.cells.values()))
        newloc = self.neighbors()[idx]
        celllocs = np.vstack([celllocs, newloc])
        if -1 in celllocs[:, 0]: # oob in x:
            celllocs[:, 0] = celllocs[:, 0] + 1
        if -1 in celllocs[:, 1]: # oob in y:
            celllocs[:, 1] = celllocs[:, 1] + 1
        # Getting rid of the negative coordinates.
        sortorder = np.lexsort((celllocs[:, 0], celllocs[:, 1]))
        self.cells = {i: loc.tolist() for i, loc in enumerate(celllocs[sortorder])}
        self.seq.append(idx)
        self.len += 1
    
    def growseq(self, seq):
        '''Grows self with seq of neighbor idxs.'''
        for num in seq:
            self.grow(num)
        self.seq.extend(seq)
    
    def get_array(self):
        array = np.zeros((self.get_max_x() + 1, self.get_max_y() + 1))
        inverted = {tuple(loc): i for i, loc in self.cells.items()}
        for key in list(inverted.keys()):
            array[key[0], key[1]] = inverted[key]
        return array
    
    def inc(self, value):
        '''Increments value.'''
        array = self.get_array()
        loc = self.cells[value]
        col = array[loc[0]]
        row = array[:, loc[1]]
        return (np.sum(col) + np.sum(row)) % self.len
    
    def add(self, a, b):
        '''Adds a and b.'''
        while b > 0:
            a = self.inc(a)
            b -= 1
        return a
    
    def get_calcs(self):
        calcs = []
        for a in range(len(self)):
            for b in range(len(self)):
                calcs.append({(a, b):self.add(a, b)})
        return calcs
    
    def check_commutativity(self):
        flag = True
        for a in range(len(self)):
            for b in range(a):
                flag = (self.add(a,b) == self.add(b,a)) and flag
        return flag
    
    def check_associativity(self):
        flag = True
        for a in range(len(self)):
            for b in range(len(self)):
                for c in range(len(self)):
                    flag = (self.add(self.add(a,b),c) == self.add(a,self.add(b,c))) and flag
        return flag

    def check_identity(self):
        flag = False
        identity = -1
        for i in range(len(self)):
            flags = [((self.add(a, i) == a) and (self.add(i, a) == a)) for a in range(len(self))]
            if all(flags):
                if identity == -1:
                    flag = True
                    identity = i
                else:
                    flag = False
                    identity = -1
                    break
        return flag, identity
    
    def check_invertible(self):
        if not self.check_identity()[0]:
            return False, [-1] * len(self)
        identity = self.check_identity()[1]
        inverses = []
        flag = True
        for a in range(len(self)):
            inverse = [((self.add(a, i) == identity) and (self.add(i, a) == identity)) for i in range(len(self))]
            if inverse.count(True) == 1:
                inverses.append(inverse.index(True))
            else:
                inverses.append(-1)
                flag = False
        return flag, inverses
    
    def check_group(self):
        return self.check_associativity() and self.check_invertible()[0]
    
    def check_standardness(self):
        # Note: I use "standard" to refer to an operation that is both associative and commutative, like addition or multiplication.
        return self.check_associativity() and self.check_commutativity()
    
    def check_abelian_group(self):
        return self.check_group() and self.check_commutativity()
    
    def __len__(self):
        return self.get_len()
    
    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.cells == other.cells
    
    def __and__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.seq == other.seq
    
    def __gt__(self, other):
        if not isinstance(other, Piece):
            return NotImplemented
        if self.len < other.len:
            return True
        elif self.len > other.len:
            return False
        else:
            sseq = self.seq
            oseq = other.seq
            for i in range(len(sseq) - 1, -1, -1):
                if sseq[i] < oseq[i]:
                    return True
                elif sseq[i] > oseq[i]:
                    return False
            return False

class PieceClass():
    '''Class of pieces.'''
    def __init__(self, pieces=None):
        '''Initializes self.'''
        if pieces:
            self.pieces = pieces
        else:
            self.pieces = np.array([])
    
    def __add__(self, other):
        '''Returns the pieces of self, with other.'''
        if isinstance(other, Iterable):
            return np.unique(np.append(self.pieces, np.array(other)))
        else:
            return np.unique(np.append(self.pieces, np.array([other])))

    def __str__(self):
        '''Turns self into a string.'''
        string = "\n--------\n".join(str(p) for p in self.pieces)
        print(f"String {string}")
        return string


    def start(self):
        '''Adds the simplest Piece() to the self.'''
        self.pieces = self + Piece()

PC = PieceClass()
PC.start()
PC.start()
PC = PC + Piece([1, 1, 1, 1, 2, 2, 1])
print(str(PC))