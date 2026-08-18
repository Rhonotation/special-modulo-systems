from __future__ import annotations
import numpy as np
from functools import total_ordering
from collections.abc import Iterable

@total_ordering
class Piece():
    def __init__(self, seq=None, tag=None):
        '''Initializes self.'''
        self.cells = {0:[0,0]}
        self.len = 1
        self.seq = []
        if seq:
            self.growseq(seq)
        if tag != None:
            self.tag = tag
        else:
            self.tag = {}
        self.calcs = self.get_calcs()

    def __str__(self):
        '''Turns self into a string.'''
        stringlst = np.array([[" " for _ in range(self.get_max_x() + 1)] for _ in range(self.get_max_y() + 1)])
        for num, cell in self.cells.items():
            stringlst[cell[1], cell[0]] = str(num) # we index y before x
        string = np.apply_along_axis(lambda x: "\t".join(x), 1, stringlst)
        return "\n".join(string)
    
    def classify(self, traits=None, recalc=False):
        """
        Compute only the requested traits.
        traits: a set of trait names to compute. If None, compute all.
        recalc: if True, recompute even if cached.
        """

        # All possible traits
        ALL_TRAITS = {
            'size', 'commutative', 'associative',
            'has_identity', 'identity',
            'invertible', 'inverses',
            'standard', 'group', 'abelian'
        }

        # Default: compute everything
        if traits is None:
            traits = ALL_TRAITS
        else:
            traits = set(traits)

        # If cached and not forced to recalc, return only requested traits
        if (not recalc) and (self.tag != {}):
            return {t: self.tag[t] for t in traits}

        # Otherwise compute missing traits
        tag = {}

        if 'size' in traits:
            tag['size'] = self.get_len()

        if 'commutative' in traits or 'standard' in traits or 'abelian' in traits:
            tag['commutative'] = bool(self.check_commutativity())

        if 'associative' in traits or 'standard' in traits or 'group' in traits or 'abelian' in traits:
            tag['associative'] = bool(self.check_associativity())

        if 'has_identity' in traits or 'identity' in traits or 'invertible' in traits or 'group' in traits or 'abelian' in traits:
            identity_info = self.check_identity()
            tag['has_identity'] = identity_info[0]
            tag['identity'] = identity_info[1]

        if 'invertible' in traits or 'inverses' in traits or 'group' in traits or 'abelian' in traits:
            invertibility_info = self.check_invertible()
            tag['invertible'] = invertibility_info[0]
            tag['inverses'] = invertibility_info[1]

        if 'standard' in traits:
            tag['standard'] = tag.get('commutative', self.check_commutativity()) and \
                            tag.get('associative', self.check_associativity())

        if 'group' in traits:
            tag['group'] = tag.get('associative', self.check_associativity()) and \
                        tag.get('invertible', self.check_invertible()[0])

        if 'abelian' in traits:
            tag['abelian'] = tag.get('group', self.check_group()) and \
                            tag.get('commutative', self.check_commutativity())

        # Cache full tag (not just requested subset)
        # This ensures future calls are fast
        self.tag.update(tag)

        # Return only requested traits
        return {t: self.tag[t] for t in traits}

    def satisfy(self, traits:dict):
        needed = set(traits.keys())
        computed = self.classify(traits=needed)
        for trait, value in traits.items():
            if computed[trait] != value:
                return False
        return True

    def copy(self):
        '''Returns a copy of self.'''
        return Piece(self.seq, self.tag)
    
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
        celllocs = np.array(list(self.cells.values()))

        # Generate all 4-direction neighbors in one vectorized operation
        offsets = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        neighborlist = celllocs[:, None, :] + offsets[None, :, :]
        neighborlist = neighborlist.reshape(-1, 2)

        # Remove existing cells
        current = set(map(tuple, celllocs))
        mask = np.array([tuple(n) not in current for n in neighborlist])
        neighborlist = neighborlist[mask]

        # Unique + sorted
        neighborlist = np.unique(neighborlist, axis=0)
        neighborlist = neighborlist[np.lexsort((neighborlist[:,0], neighborlist[:,1]))]

        return neighborlist.tolist()
    
    def grow(self, idx, recalc=True):
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
        self.tag = {}
        if recalc:
            self.calcs = self.get_calcs()
        return self
    
    def growseq(self, seq):
        '''Grows self with seq of neighbor idxs.'''
        if len(seq) > 0:
            for num in seq:
                self.grow(num, recalc=False)
            self.calcs = self.get_calcs()
    
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
        return (np.sum(col) + np.sum(row)) % self.get_len()
    
    def add(self, a, b):
        '''Adds a and b.'''
        while b > 0:
            a = self.inc(a)
            b -= 1
        return a
    
    def get_incs(self):
        incs = np.array([self.inc(a) for a in range(self.len)])
        return incs
    
    def get_calcs(self):
        calcs = np.zeros((self.len, self.len), dtype=int)
        calcs[0] = np.arange(self.len)
        if self.len > 1:
            calcs[1] = self.get_incs()
        for i in range(2, self.len):
            calcs[i] = np.array([self.inc(calcs[i-1][b]) for b in range(self.len)])
        return calcs

    def check_commutativity(self):
        return np.all(self.calcs == self.calcs.T)
    
    def check_associativity(self):
        calcs = self.calcs
        for a in range(len(self)):
            for b in range(len(self)):
                for c in range(len(self)):
                    if calcs[calcs[a,b],c] != calcs[a,calcs[b,c]]:
                        return False
        return True

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
        return len(self.seq)
    
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
    def __init__(self, pieces=[]):
        '''Initializes self.'''
        if len(pieces) > 0:
            self.pieces = np.array(pieces)
        else:
            self.pieces = np.array([])

    def __getitem__(self, key:int|Iterable[int]|slice):
        if isinstance(key, slice):
            return self.pieces[key]
        else:
            if isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
                return self.pieces[np.array(key)]
            else:
                return self.pieces[key]

    def __setitem__(self, key:int|slice, val:Piece|Iterable[Piece]):
        self.pieces[key] = val
    
    def __add__(self, other:Piece|Iterable[Piece]| PieceClass, detectunique = False): # type: ignore
        '''Returns the pieces of self, with other.'''
        if isinstance(other, PieceClass):
            if detectunique:
                returnval = PieceClass(np.unique(np.append(self.pieces, other.pieces)))
            else:
                returnval = PieceClass(np.append(self.pieces, other.pieces))
        elif isinstance(other, Piece):
            # We have a piece!
            if other in self:
                returnval = PieceClass(self.pieces)
            else:
                returnval = PieceClass(np.append(self.pieces, other))
        elif isinstance(other, Iterable):
            if detectunique:
                returnval = PieceClass(np.unique(np.append(self.pieces, other)))
            else:
                returnval = PieceClass(np.append(self.pieces, other))
        try:
            return returnval
        except BaseException:
            raise TypeError(f"other must be a Piece(), an Iterable of Piece()s, or a PieceClass(), but it is of {str(type(other))[1:-1]}.")

    def __iadd__(self, other:Piece|Iterable[Piece]| PieceClass ): # type: ignore
        self.pieces = (self + other).pieces
        return self

    def __reversed__(self):
        return iter(np.flip(self.pieces))

    def __str__(self):
        '''Turns self into a string.'''
        string = "\n--------\n".join([str(p) for p in self.pieces])
        return string

    def __len__(self):
        '''Gets the length of self.'''
        return len(self.pieces)

    def __iter__(self):
        return iter(self.pieces)

    def __contains__(self, item):
        return item in self.pieces

    def __delitem__(self, key):
        self.pieces = np.append(self.pieces[:key], self.pieces[key + 1:])

    def __repr__(self):
        return str([str(piece) for piece in self.pieces])

    def __eq__(self, other):
        if not isinstance(other, PieceClass):
            return False
        if len(self.pieces) != len(other.pieces):
            return False
        return all(self.pieces[i] == other.pieces[i] for i in range(len(self.pieces)))

    def get_grow(self):
        '''Gets pieces to grow self with from its longest values.'''
        lengths = np.array([p.get_len() for p in self.pieces])
        growfromlen = np.max(lengths)
        growpieces = self.pieces[np.where(lengths == growfromlen)]
        grownpieces = PieceClass()
        nsum = 0
        for piece in growpieces:
            neighborcount = len(piece.neighbors())
            nsum += neighborcount
            for i in range(neighborcount):
                grownpieces += piece.copy().grow(i)
        return grownpieces

    def grow(self, iters=1):
        '''Grows self iters times.'''
        for _ in range(iters):
            self += self.get_grow()
        return self

    def start(self):
        '''Adds the simplest Piece() to the self.'''
        self += Piece()

    def random(self):
        '''Returns a random element of self.'''
        return np.random.choice(self.pieces)

    def query(self, attribute_query:dict):
        '''Takes an attribute_query, which is a dictionary of wanted traits and their values.
        Then, returns a PieceClass() with every piece in self satisfying those.'''
        satisfy_pieces = PieceClass([p for p in self.pieces if p.satisfy(attribute_query)])
        return satisfy_pieces

    def get_seqs(self):
        '''Gets the sequences of the Piece()s of self.'''
        seqs = [p.seq for p in self.pieces]
        return seqs

    def get_matrices(self):
        '''Gets the matrices of the Piece()s of self.'''
        matrices = [p.get_array().tolist() for p in self.pieces]
        return matrices

    def repgrow(self, iters=1):
        for _ in range(iters):
            self.pieces = self.get_grow().pieces
        return self