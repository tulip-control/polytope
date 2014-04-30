# -*- coding: utf-8 -*-
#
# Copyright (c) 2011-2014 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
#
#
#  Acknowledgement:
#  The overall structure of this library and the functions in the list
#  below are taken with permission from:
#
#  M. Kvasnica, P. Grieder and M. Baotić,
#  Multi-Parametric Toolbox (MPT),
#  http://control.ee.ethz.ch/~mpt/
#
#  mldivide
#  region_diff
#  extreme
#  envelope
#  is_convex
#  bounding_box
#  intersect2
#  projection_interhull
#  projection_exthull
#
"""
Computational geometry module for polytope computations.
Suggested abbreviation:

>>> import tulip.polytope as pc
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from cvxopt import matrix, solvers

try:
    import matplotlib as mpl
    from tulip.graphics import newax, dom2vec
except Exception, e:
    logger.error(e)
    mpl = None
    newax = None

from .quickhull import quickhull
from .esp import esp

# Find a lp solver to use
try:
    import cvxopt.glpk
    lp_solver = 'glpk'
except:
    logger.warn("GLPK (GNU Linear Programming Kit) solver for CVXOPT not found, "
           "reverting to CVXOPT's own solver. This may be slow")
    lp_solver = None

# Hide optimizer output
solvers.options['show_progress'] = False
solvers.options['LPX_K_MSGLEV'] = 0

# Nicer numpy output
np.set_printoptions(precision=5, suppress = True)

# global default absolute tolerance,
# to enable changing it code w/o passing arguments,
# so that magic methods can still be used
ABS_TOL = 1e-7

class ConvexPolytope(object):
    """Convex polytope in hyperplane representation:
    
      - C{A}: hyperplane normals
        @type A: numpy.ndarray
      
      - C{b}: hyperplane offsets
        @type b: numpy.ndarray
    
    The following are set on first attribute access:
    
      - Chebyshev center and radius:
        
        - C{x}
        - C{r}
      
      - C{dim}: dimension
      - C{volume}
      - C{bounding_box}
      - C{vertices}
      
    and if in minimal representation,
    after applying L{_reduce}:
    
      - C{minrep}: is set to C{True}
    
    See Also
    ========
    L{Polytope}
    """
    def __init__(self,
        A=None, b=None,
        minrep = False, normalize=True,
        abs_tol=ABS_TOL
    ):
        """Instantiate L{ConvexPolytope}.
        
          - C{normalize}: if True (default),
              then normalize given C{A}, C{b}
              otherwise don't modify them.
        """
        if A is None:
            A = np.array([])
        
        if b is None:
            b = np.array([])
        
        self.A = A.astype(float)
        self.b = b.astype(float).flatten()
        if A.size > 0 and normalize:
            # Normalize
            Anorm = np.sqrt(np.sum(A*A,1)).flatten()     
            pos = np.nonzero(Anorm > 1e-10)[0]
            self.A = self.A[pos, :]
            self.b = self.b[pos]
            Anorm = Anorm[pos]           
            mult = 1/Anorm
            for i in xrange(self.A.shape[0]):
                self.A[i,:] = self.A[i,:]*mult[i]
            self.b = self.b.flatten()*mult
        self.minrep = minrep
        self._x = None
        self._r = 0
        self._bbox = None
        self._fulldim = None
        self._volume = None
        self._vertices = None
        self._abs_tol = abs_tol

    def __str__(self):
        """Return pretty-formatted H-representation of polytope(s).
        """
        try:
            output = 'Single polytope \n  '
            A = self.A
            b = self.b
            A_rows = str(A).split('\n')
            if len(b.shape) == 1:
                # If b is just an array, rather than column vector,
                b_rows = str(b.reshape(b.shape[0], 1)).split('\n')
            else:
                # Else, b is a column vector.
                b_rows = str(b).split('\n')
            mid_ind = (len(A_rows)-1)/2
            spacer = ' |    '
                    
            if mid_ind > 1:
                output += '\n  '.join([A_rows[k]+spacer+b_rows[k] \
                                        for k in xrange(mid_ind)]) + '\n'
            elif mid_ind == 1:
                output += A_rows[0]+spacer+b_rows[0] + '\n'
            else:
                output += ''
            
            output += '  ' + A_rows[mid_ind]+' x <= '+b_rows[mid_ind]
            
            if mid_ind+1 < len(A_rows)-2:
                output += '\n' +'\n  '.join([
                    A_rows[k]+spacer+b_rows[k]
                    for k in xrange(mid_ind+1, len(A_rows)-1)
                ])
            elif mid_ind+1 == len(A_rows)-2:
                output += '\n  ' + A_rows[mid_ind+1]+spacer+b_rows[mid_ind+1]
            if len(A_rows) > 1:
                output += '\n  '+A_rows[-1]+spacer[1:]+b_rows[-1]
            
            output += "\n"
            
            return output
            
        except:
            return str(self.A) + str(self.b)

    def __copy__(self):
        A = self.A.copy()
        b = self.b.copy()
        P = ConvexPolytope(A, b)
        
        self._copy_expensive_attributes(P)
        P.minrep = self.minrep
        
        return P
        
    def _copy_expensive_attributes(self, other):
        """Copy those attributes that are expensive to recompute.
        """
        other._x = self._x
        other._r = self._r
        
        other._bbox = self._bbox
        other._vertices = self._vertices
        other._fulldim = self._fulldim
        other._abs_tol = self._abs_tol
    
    def __contains__(self, point, abs_tol=ABS_TOL):
        """Return True if polytope contains point.
        
        See Also
        ========
        L{is_inside}
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        test = self.A.dot(point.flatten() ) - self.b < abs_tol
        return np.all(test)
    
    def are_inside(self, points, abs_tol=ABS_TOL):
        test = self.A.dot(points) -self.b[:,np.newaxis] < abs_tol
        return np.all(test, axis=0)
    
    def __nonzero__(self):
        return bool(self.volume > 0)
    
    def intersection(self, other, abs_tol=ABS_TOL):
        """Return intersection with ConvexPolytope or Polytope.
        
        @type other: L{Polytope}.
        
        @rtype: L{ConvexPolytope} or L{Polytope}
        """
        if not isinstance(other, ConvexPolytope):
            msg = 'Polytope intersection defined only'
            msg += ' with other Polytope. Got instead: '
            msg += str(type(other) )
            raise Exception(msg)
        
        if not self.is_fulldim() or not other.is_fulldim():
            return ConvexPolytope()
        
        if self.dim != other.dim:
            raise Exception("polytopes have different dimension")
        
        iA = np.vstack([self.A, other.A])
        ib = np.hstack([self.b, other.b])
        
        return ConvexPolytope(iA, ib).reduction(abs_tol=abs_tol)
    
    def copy(self):
        """Return copy of this Polytope.
        """
        return self.__copy__()
        
    @classmethod
    def from_box(cls, intervals=[]):
        """Class method for easy construction of hyperrectangles.
        
        @param intervals: intervals [xi_min, xi_max],
            the cross-product of which defines the polytope
            as an N-dimensional hyperrectangle
        @type intervals: [ndim x 2] numpy array or
            list of lists::

                [[x0_min, x0_max],
                 [x1_min, x1_max],
                 ...
                 [xN_min, xN_max]]
                
        @return: hyperrectangle defined by C{intervals}
        @rtype: L{Polytope}
        """
        if not isinstance(intervals, np.ndarray):
            try:
                intervals = np.array(intervals)
            except:
                raise Exception('Polytope.from_box:' +
                    'intervals must be a numpy ndarray or ' +
                    'convertible as arg to numpy.array')
        
        if intervals.ndim != 2:
            raise Exception('Polytope.from_box: ' +
                'intervals must be 2 dimensional')
        
        n = intervals.shape
        if n[1] != 2:
            raise Exception('Polytope.from_box: ' +
                'intervals must have 2 columns')
        
        n = n[0]
        
        # a <= b for each interval ?
        if (intervals[:,0] > intervals[:,1]).any():
            msg = 'Polytope.from_box: '
            msg += 'Invalid interval in from_box method.\n'
            msg += 'First element of an interval must'
            msg += ' not be larger than the second.'
            raise Exception(msg)
        
        A = np.vstack([np.eye(n),-np.eye(n)])
        b = np.hstack([intervals[:,1], -intervals[:,0] ])
        
        return cls(A, b, minrep=True)
    
    def projection(self, dim, solver=None,
                abs_tol=ABS_TOL, verbose=0):
        """Return Polytope projection on selected subspace.
        
        For usage details see function: L{projection}.
        """
        return _projection(self, dim, solver, abs_tol, verbose)
    
    def scale(self, factor):
        """Multiply polytope by scalar factor.
        
        A x <= b, becomes: A x <= (factor * b)
        
        @type factor: float
        """
        self.b = factor * self.b
    
    def reduction(self, abs_tol=ABS_TOL):
        p = _reduce(self, abs_tol=abs_tol)
        self._copy_expensive_attributes(p)
        return p
    
    @property
    def dim(self):
        """Return Polytope dimension.
        """
        try:
            return np.shape(self.A)[1]
        except:
            return 0.0
    
    @property
    def volume(self):
        if self._volume is None:
            self._volume = _volume(self)
        return self._volume
    
    @property
    def r(self):
        self._cheby
        return self._r
    
    @property
    def x(self):
        self._cheby
        return self._x
    
    @property
    def _cheby(self):
        x, r = _cheby_ball(self)
        
        self._x = x
        self._r = r
        
        return x, r
    
    @property
    def bounding_box(self):
        """Wrapper of L{polytope.bounding_box}.
        
        Computes the bounding box on first call.
        """
        if self._bbox is None:
            self._bbox = _bounding_box(self)
        return self._bbox
    
    @property
    def vertices(self):
        """Return extreme points of convex polytope.
        
        Computed on first access.
        """
        if self._vertices is None:
            self._vertices = _extreme(self)
        return self._vertices
    
    def is_fulldim(self, abs_tol=ABS_TOL):
        if self._fulldim is None:
            self._fulldim = self.r > abs_tol
        return self._fulldim
    
    def is_empty(self):
        """Check True if A is either None or an empty matrix.
        """
        try:
            return len(self.A) == 0
        except:
            return True
    
    def plot(self, ax=None, color=None,
             hatch=None, alpha=1.0):
        if color is None:
            color = np.random.rand(3)
        
        if newax is None:
            logger.warn('newax not imported. No Polytope plotting.')
            return
        
        if ax is None:
            ax, fig = newax()
        
        if not self.is_fulldim():
            logger.error("Cannot plot empty polytope")
            return ax
        
        if self.dim != 2:
            logger.error("Cannot plot polytopes of dimension larger than 2")
            return ax
        
        poly = _get_patch(
            self, facecolor=color, hatch=hatch,
            alpha=alpha, linestyle='dashed', linewidth=3,
            edgecolor='black'
        )
        ax.add_patch(poly)
        
        return ax
    
    def text(self, txt, ax=None, color='black'):
        """Plot text at chebyshev center.
        """
        _plot_text(self, txt, ax, color)

class Polytope(object):
    """Iterable container of convex polytopes.
    
    Attributes:
    
      - C{props}: set of labels annotating the C{Polytope}.
      
    The following attributes are computed on first access:
    
      - C{dim}: dimension
      - C{volume}
      - C{x}: coordinates Chebyshev center
      - C{r}: Chebyshev radius
      
      - C{bounding_box}
      - C{envelope}: convex envelope
    
    See Also
    ========
    L{ConvexPolytope}
    """
    def __init__(self, convex_polytopes=None,
                 props=None, abs_tol=ABS_TOL):
        if convex_polytopes is None:
            convex_polytopes = []
        if props is None:
            props = set()
        
        if isinstance(convex_polytopes, str):
            # Hack to be able to use the Polytope class also for discrete
            # problems.
            self._polytopes = convex_polytopes
            self.props = set(props)
        else:
            if isinstance(convex_polytopes, Polytope):
                p = convex_polytopes
                dim = p[0].dim
                for poly in p:
                    if poly.dim != dim:
                        raise Exception("Polytope error:"
                            " Polytopes must be of same dimension!")                    
            
            self._polytopes = convex_polytopes[:]
            
            for poly in convex_polytopes:
                if poly.is_empty():
                    self._polytopes.remove(poly)
            
            self.props = set(props)
            self._bbox = None
            self._fulldim = None
            self._volume = None
            self._x = None
            self._r = None
            self._is_empty = None
            self._envelope = None
            self._abs_tol = abs_tol
    
    def __iter__(self):
        return iter(self._polytopes)
    
    def __getitem__(self, key):
        return self._polytopes[key]
        
    def __str__(self):
        output = ''
        for i, poly in enumerate(self):
            output += '\t Polytope number ' + str(i+1) + ':\n'
            
            poly_str = str(poly)
            poly_str = poly_str.replace('\n', '\n\t\t')
            
            output += '\t ' + poly_str + '\n'
        output += '\n'
        return output  
        
    def __len__(self):
        return len(self._polytopes)
    
    def __contains__(self, point, abs_tol=ABS_TOL):
        """Return True if Polytope contains point.
        
        See Also
        ========
        L{is_inside}
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        
        for poly in self:
            if poly.__contains__(point, abs_tol):
                return True
        return False
    
    def __eq__(self, other):
        return self <= other and other <= self
    
    def __ne__(self, other):
        return not self == other
    
    def __le__(self, other):
        return _is_subset(self, other)
    
    def __ge__(self, other):
        return _is_subset(other, self)
    
    def __add__(self, other):
        """Return union with C{other}.
        
        Applies convex simplification if possible.
        To turn off this check,
        use Region.union
        
        @type other: L{Polytope}
        
        @rtype: L{Polytope}
        """
        return _union(self, other, check_convex=True)
    
    def __nonzero__(self):
        return bool(self.volume > 0)
    
    def union(self, other, check_convex=False):
        """Return union with C{other}.
        
        For usage see function union.
        
        @type other: L{Polytope}
        
        @rtype: L{Polytope}
        """
        return _union(self, other, check_convex)
    
    def __sub__(self, other):
        """Return set difference with C{other}.
        
        @type other: L{Polytope}
        
        @rtype: L{Polytope}
        """
        return _mldivide(self, other)
    
    def diff(self, other):
        """Return set difference with C{other}.
        
        @type other: L{Polytope}
        
        @rtype: L{Region}
        """
        return _mldivide(self, other)
        
    def __and__(self, other):
        """Return intersection with C{other}.
        
        Absolute tolerance 1e-7 used.
        To select the absolute tolerance use
        method Region.intersect
        
        @type other: L{Region}
        
        @rtype: L{Region}
        """
        return self.intersection(other)
    
    def intersection(self, other, abs_tol=ABS_TOL):
        """Return intersection with C{other}.
        
        @type other: C{Polytope}.
        
        @rtype: L{Polytope}
        """
        P = Polytope()
        for poly0 in self:
            for poly1 in other:
                isect = poly0.intersection(poly1, abs_tol)
                rp, xp = isect.cheby
            
                if rp > abs_tol:
                    P = _union(P, isect, check_convex=True)
        return P
    
    def projection(self, dim, solver=None, abs_tol=ABS_TOL):
        """Return Polytope projection on selected subspace.
        
        For usage details see function: L{projection}.
        """
        proj = Polytope()
        for poly in self:
            proj += poly.projection(dim, solver, abs_tol)
        return proj
    
    def __copy__(self):
        """Return copy of this Polytope.
        """
        r = Polytope(self._polytopes[:], props=self.props.copy() )
        
        self._copy_expensive_attributes(r)
        return r
        
    def _copy_expensive_attributes(self, other):
        other._bbox = self._bbox
        other._fulldim = self._fulldim
        other._volume = self._volume
        other._x = self._x
        other._r = self._r
        other._envelope = self._envelope
        other._is_empty = self._is_empty
        other._abs_tol = self._abs_tol
    
    def copy(self):
        """Return copy of this L{Polytope}.
        """
        return self.__copy__()
    
    def reduction(self, abs_tol=ABS_TOL):
        new_polys = []
        for poly in self:
            red = poly.reduction(abs_tol=abs_tol)
            
            if red.is_fulldim():
                new_polys.append(red)
        
        p = Polytope(new_polys, self.props)
        
        self._copy_expensive_attributes(p)
        
        return p
    
    @property
    def dim(self):
        """Return L{Polytope} dimension.
        """
        return np.shape(self[0].A)[1]
    
    @property
    def volume(self):
        if self._volume is None:
            self._volume = 0.0
            for poly in self:
                self._volume += poly.volume
        return self._volume
    
    @property
    def r(self):
        self._cheby
        return self._r
    
    @property
    def x(self):
        self._cheby
        return self._x
    
    def _cheby(self):
        if self._x is None or self._r is None:
            maxr = 0
            maxx = None
            for poly in self:
                r = poly.r
                
                if r > maxr:
                    maxr = r
                    maxx = poly.x
            
            self._x = maxx
            self._r = maxr
        
        return self._x, self._r
    
    @property
    def bounding_box(self):
        """Compute the bounding box on first call.
        """
        # For regions, calculate recursively for each
        # convex polytope and take maximum
        if self._bbox is None:
            lenP = len(self)
            dimP = self.dim
            alllower = np.zeros([lenP,dimP])
            allupper = np.zeros([lenP,dimP])
            
            for i, poly in enumerate(self):
                ll, uu = poly.bounding_box
                alllower[i, :] = ll.T
                allupper[i, :] = uu.T
            
            l = np.zeros([dimP,1])
            u = np.zeros([dimP,1])
            
            for i in xrange(0,dimP):
                l[i] = min(alllower[:, i])
                u[i] = max(allupper[:, i])
            
            self._bbox = (l, u)
        return self._bbox
    
    def envelope(self, abs_tol=ABS_TOL):
        """Return envelope of a L{Polytope}.
        """
        if self._envelope is None:
            self._envelope = _envelope(self, abs_tol=ABS_TOL)
        return self._envelope
    
    def is_empty(self):
        if self._is_empty is None:
            self._is_empty = False
            for poly in self:
                if poly.is_empty():
                    self._is_empty = True
                    break
        return self._is_empty
    
    def is_fulldim(self):
        """Return True if there exist interior points.
        """
        if self._fulldim is None:
            for poly in self:
                if poly.is_fulldim():
                    self._fulldim = True
                    break
        return self._fulldim
    
    def is_adjacent(self, other, overlap=False, abs_tol=None):
        if not isinstance(other, Polytope):
            raise TypeError('other must be a Polytope.')
        
        if abs_tol is None:
            abs_tol = self._abs_tol
        
        for p0 in self:
            for p1 in other:
                if _is_adjacent(p0, p1, overlap=overlap, abs_tol=abs_tol):
                    return True
        return False
    
    def is_interior(self, other, abs_tol=ABS_TOL):
        """Return True if C{other} is strictly in the interior of C{self}.
        
        Checks if C{other} enlarged by C{abs_tol}
        is a subset of C{self}.
        
        @type other: L{Polytope}
        
        @rtype: bool
        """
        for p in other:
            A = p.A.copy()
            b = p.b.copy() + abs_tol
            
            dummy = ConvexPolytope(A, b)
            
            if not dummy <= self:
                return False
        return True
    
    def plot(self, ax=None, color=None,
             hatch=None, alpha=1.0):
        if color is None:
            color = np.random.rand(3)
        
        if newax is None:
            logger.warn('pyvectorized not found. No plotting.')
            return None
        
        #TODO optional arg for text label
        if not self.is_fulldim():
            logger.error("Cannot plot empty region")
            return
        
        if self.dim != 2:
            logger.error("Cannot plot region of dimension larger than 2")
            return
        
        if ax is None:
            ax, fig = newax()
        
        for poly2 in self:
            # TODO hatched polytopes in same region
            poly2.plot(ax, color=color, hatch=hatch, alpha=alpha)
        
        return ax
    
    def text(self, txt, ax=None, color='black'):
        """Plot text at chebyshev center.
        """
        _plot_text(self, txt, ax, color)
      
    def is_convex(self, abs_tol=ABS_TOL):
        """Check if a region is convex.
        
        @type reg: L{Polytope}
        
        @return: result,envelope: result indicating if convex.
            If found to be convex the envelope describing
            the convex polytope is returned.
        """
        # is this check a bug ?
        if not self.is_fulldim():
            return True
        
        outer = _envelope(self)
        if outer.is_empty():
            # Probably because input polytopes
            # were so small and ugly...
            return False,None
    
        Pl,Pu = self.bounding_box
        Ol,Ou = outer.bounding_box
        
        bboxP = np.hstack([Pl,Pu])
        bboxO = np.hstack([Ol,Ou])
        
        if sum(abs(bboxP[:,0] - bboxO[:,0]) > abs_tol) > 0 or \
        sum(abs(bboxP[:,1] - bboxO[:,1]) > abs_tol) > 0:
            return False,None
        if outer.diff(self).is_fulldim():
            return False,None
        else:
            return True,outer

def _is_subset(small, big, abs_tol=ABS_TOL):
    """Return True if small \subseteq big.
    
    @type small: L{Polytope}
    @type big: L{Polytope}
    
    @rtype: bool
    """
    for x in [small, big]:
        if not isinstance(x, Polytope):
            msg = 'Not a Polytope, got instead:\n\t'
            msg += str(type(x))
            raise TypeError(msg)
    
    diff = small.diff(big)
    volume = diff.volume
    
    if volume < abs_tol:
        return True
    else:
        return False

def _reduce(poly, nonEmptyBounded=1, abs_tol=ABS_TOL):  
    """Removes redundant inequalities in the hyperplane representation.
    
    Using algorithm from:
        http://www.ifor.math.ethz.ch/~fukuda/polyfaq/node24.html
    by solving one LP for each facet.

    Warning:
      - nonEmptyBounded == 0 case is not tested much.
    
    @type poly: L{ConvexPolytope}
    
    @return: Reduced L{ConvexPolytope}
    """
    if poly.minrep:
        # If polytope already in minimal representation
        return poly
        
    if not poly.is_fulldim():
        return ConvexPolytope()
    
    A_arr = poly.A
    b_arr = poly.b
    
    # Remove rows with b = inf
    keep_row = np.nonzero(poly.b != np.inf)
    A_arr = A_arr[keep_row]
    b_arr = b_arr[keep_row]
    
    neq = np.shape(A_arr)[0]
    # first eliminate the linearly dependent rows
    # corresponding to the same hyperplane
    M1 = np.hstack([A_arr,np.array([b_arr]).T]).T
    M1row = 1/np.sqrt(np.sum(M1**2,0))
    M1n = np.dot(M1,np.diag(M1row)) 
    M1n = M1n.T
    keep_row = []
    for i in xrange(neq):
        keep_i = 1
        for j in xrange(i+1,neq):
            if np.dot(M1n[i].T,M1n[j])>1-abs_tol:
                keep_i = 0
        if keep_i:
            keep_row.append(i)
    
    A_arr = A_arr[keep_row]
    b_arr = b_arr[keep_row]
    neq, nx = A_arr.shape
    
    if nonEmptyBounded:
        if neq<=nx+1:
            return ConvexPolytope(A_arr,b_arr)
    
    # Now eliminate hyperplanes outside the bounding box
    if neq>3*nx:
        lb, ub = ConvexPolytope(A_arr,b_arr).bounding_box
        #cand = -(np.dot((A_arr>0)*A_arr,ub-lb)
        #-(b_arr-np.dot(A_arr,lb).T).T<-1e-4)
        cand = -(
            np.dot((A_arr>0)*A_arr,ub-lb)
            -(np.array([b_arr]).T-np.dot(A_arr,lb))
            < -1e-4
        )
        A_arr = A_arr[cand.squeeze()]
        b_arr = b_arr[cand.squeeze()]
    
    neq, nx = A_arr.shape
    if nonEmptyBounded:
        if neq<=nx+1:
            return ConvexPolytope(A_arr,b_arr)
         
    del keep_row[:]
    for k in xrange(A_arr.shape[0]):
        f = -A_arr[k,:]
        G = A_arr
        h = b_arr
        h[k] += 0.1
        sol=solvers.lp(
            matrix(f), matrix(G), matrix(h),
            None, None, lp_solver
        )
        h[k] -= 0.1
        if sol['status'] == "optimal":
            obj = -sol['primal objective'] - h[k]
            if obj > abs_tol:
                keep_row.append(k)
        elif sol['status'] == "dual infeasable":
            keep_row.append(k)
        
    polyOut = ConvexPolytope(A_arr[keep_row],b_arr[keep_row])
    polyOut.minrep = True
    return polyOut

def _union(p0, p1, check_convex=False):
    """Compute the union of polytopes or regions
    
    @type polyreg1: L{Polytope}
    @type polyreg2: L{Polytope}
    @param check_convex: if True, look for convex unions and simplify
    
    @return: region of non-overlapping polytopes describing the union
    """
    #logger.debug('union')
    
    assert(isinstance(p0, Polytope) )
    assert(isinstance(p1, Polytope) )
    
    if p0.is_empty():
        return p1
    if p1.is_empty():
        return p0
    
    if check_convex:
        s1 = p0.intersection(p1)
        
        if s1.is_fulldim():
            s2 = p1.diff(p0)
            s3 = p0.diff(p1)
        else:
            s2 = p0
            s3 = p1
        
        s = [s1, s2, s3]
    else:
        s = [p0, p1]
    
    polys = []
    for p in s:
        assert(isinstance(p, Polytope) )
        
        for poly in p:
            assert(isinstance(poly, ConvexPolytope) )
            
            if not poly.is_empty():
                polys.append(poly)
    
    if not check_convex:
        return Polytope(polys)
    
    if len(polys) == 1:
        return Polytope(polys)
    
    # check convexity by incrementally adding polytopes
    final = []
    while polys:
        cvx = [polys[0]]
        
        for p in polys[1:]:
            cvx.append(p)
            is_conv, env = Polytope(cvx).is_convex()
            
            # skip p if it breaks convexity
            if not is_conv:
                cvx.remove(p)
        
        # remove those that fit together convexly
        for p in cvx:
            polys.remove(p)
        
        # glue them together
        cvxpoly = Polytope(cvx).envelope().reduction()
        
        if cvxpoly.is_empty():
            continue
        
        assert(len(cvxpoly) == 1)
        final.append(cvxpoly[0])
    
    return Polytope(final)

def _cheby_ball(poly):
    """Calculate the Chebyshev radius and center for a polytope.

    If input is a region the largest Chebyshev ball is returned.
    
    N.B., this function will return whatever it finds in attributes
    r and x if not None, without (re)computing the Chebyshev ball.
    
    Example (low dimension):
    
    r1,x1 = cheby_ball(P, [1]) calculates the center and half the
    length of the longest line segment along the first coordinate axis
    inside polytope P

    @type poly1: L{Polytope}
    
    @return: rc,xc: Chebyshev radius rc (float) and center xc (numpy array)
    """
    #logger.debug('cheby ball')
    if poly.is_empty():
        return 0,None

    r = 0
    xc = None
    A = poly.A
    
    c = -matrix(np.r_[np.zeros(np.shape(A)[1]),1])
    
    norm2 = np.sqrt(np.sum(A*A, axis=1))
    G = np.c_[A, norm2]
    G = matrix(G)
    
    h = matrix(poly.b)
    sol = solvers.lp(c, G, h, None, None, lp_solver)
    if sol['status'] == "optimal":
        r = sol['x'][-1]
        if r < 0:
            return None, 0
        xc = sol['x'][0:-1]
    else:
        # Polytope is empty
        return None, 0
    
    xc = np.array(xc)
    r = np.double(r)
    
    return xc, r
    
def _bounding_box(polyreg):
    """Return smallest hyperbox containing polytope or region.
    
    @type polyreg: L{Polytope} or L{Region}
    
    @return: (l, u) where:
        
        - l = [x1min,
               x2min,
               ...
               xNmin]
        
        - u = [x1max,
               x2max,
               ...
               xNmax]
        
    @rtype:
        - l = 2d array
        - u = 2d array
    """
    # For one convex polytope, solve an optimization problem
    (m, n) = np.shape(polyreg.A)
    
    In = np.eye(n)
    l = np.zeros([n,1])
    u = np.zeros([n,1])
    
    for i in xrange(0,n):
        c = matrix(np.array(In[:,i]))
        G = matrix(polyreg.A)
        h = matrix(polyreg.b)
        sol = solvers.lp(c, G, h, None, None, lp_solver)
        if sol['status'] == "optimal":
            x = sol['x']
            l[i] = x[i]
            
    for i in xrange(0,n):
        c = matrix(-np.array(In[:,i]))
        G = matrix(polyreg.A)
        h = matrix(polyreg.b)
        sol = solvers.lp(c, G, h, None, None, lp_solver)
        if sol['status'] == "optimal":
            x = sol['x']
            u[i] = x[i]
    
    return l,u
    
def _envelope(reg, abs_tol=ABS_TOL):
    """Compute envelope of a L{Polytope}.

    The envelope is the polytope defined by all "outer" inequalities a
    x < b such that {x | a x < b} intersection P = P for all polytopes
    P in the region. In other words we want to find all "outer"
    equalities of the region.
    
    If envelope can't be computed an empty polytope is returned
    
    @type reg: L{Polytope}
    @param abs_tol: Absolute tolerance for calculations
    
    @return: Envelope of input
    @rtype: L{Polytope}
    """
    Ae = None
    be = None
    
    for poly1 in reg:
        outer_i = np.ones(poly1.A.shape[0])
        for ii in xrange(poly1.A.shape[0]):
            if outer_i[ii] == 0:
                # If inequality already discarded
                continue
            for poly2 in reg:
                # Check for each polytope
                # if it intersects with inequality ii
                if poly1 is poly2:
                    continue
                testA = np.vstack([poly2.A, -poly1.A[ii,:]])
                testb = np.hstack([poly2.b, -poly1.b[ii]])
                testP = ConvexPolytope(testA,testb)
                
                if testP.r > abs_tol:
                    # poly2 intersects with inequality ii -> this inequality
                    # can not be in envelope
                    outer_i[ii] = 0
        ind_i = np.nonzero(outer_i)[0]
        if Ae is None:
            Ae = poly1.A[ind_i,:]
            be = poly1.b[ind_i]
        else:
            Ae = np.vstack([Ae, poly1.A[ind_i,:]])
            be = np.hstack([be, poly1.b[ind_i]])
    
    ret = ConvexPolytope(Ae,be).reduction()
    if ret.is_fulldim():
        return Polytope([ret])
    else:
        return Polytope()

count = 0

def _mldivide(a, b, save=False):
    """Return set difference a \ b.
    
    @param a: L{Polytope}
    @param b: L{Polytope} or L{ConvexPolytope} to subtract
    
    @return: L{Polytope} describing the set difference
    """
    assert(isinstance(a, Polytope) )
    
    r = Polytope()
    for p in a:
        assert(isinstance(p, ConvexPolytope) )
        #assert(not is_fulldim(P.intersection(poly) ) )
        
        if isinstance(b, ConvexPolytope):
            diff = _region_diff(p, b)
        elif isinstance(b, Polytope):
            diff = p
            for q in b:
                assert(isinstance(q, ConvexPolytope) )
                
                # recurse only when necessary
                if isinstance(diff, Polytope):
                    diff = _mldivide(diff, q)
                elif isinstance(diff, ConvexPolytope):
                    diff = _region_diff(diff, q)
                else:
                    raise TypeError('diff not polytope')
        else:
            raise TypeError('b not polytope')
        
        r = r.union(diff, check_convex=True)
        
        if save:
            global count
            count = count + 1
            
            ax = diff.plot()
            ax.axis([0.0, 1.0, 0.0, 2.0])
            ax.figure.savefig('./img/Pdiff' + str(count) + '.pdf')
            
            ax = r.plot()
            ax.axis([0.0, 1.0, 0.0, 2.0])
            ax.figure.savefig('./img/P' + str(count) + '.pdf')
        
    return r
    
def _volume(poly):
    """Approximate volume of L{ConvexPolytope}.
    
    A randomized algorithm is used.
    
    @type poly: L{ConvexPolytope}
    
    @return: volume of C{poly}
    """
    if not poly.is_fulldim():
        return 0.0
    
    logger.debug('computing volume...')

    n = poly.A.shape[1]
    if n == 1:
        N = 50
    elif n == 2:
        N = 500
    elif n ==3:
        N = 3000
    else:
        N = 10000
    
    l_b, u_b = poly.bounding_box
    
    x = np.tile(l_b,(1,N)) +\
        np.random.rand(n,N) *\
        np.tile(u_b-l_b,(1,N) )
    
    aux = np.dot(poly.A, x) -\
        np.tile(
            np.array([poly.b]).T,
            (1, N)
        )
    
    aux = np.nonzero(np.all(((aux < 0)==True), 0) )[0].shape[0]
    vol = np.prod(u_b-l_b) * aux / N
    
    return vol    
            
def _extreme(poly1):
    """Compute the extreme points of a bounded convex polytope.
    
    @param poly1: convex polytope in dimension d
    @type poly1: L{ConvexPolytope}
    
    @return: array containing N vertices of C{poly1}
    @rtype: (N x d) numpy array
    """
    if isinstance(poly1, Polytope):
        raise Exception('poly1 must be a ConvexPolytope')

    V = np.array([])
    R = np.array([])
    
    poly1 = poly1.reduction() # Need to have polytope non-redundant!

    if not poly1.is_fulldim():
        return None
    
    A = poly1.A.copy()
    b = poly1.b.copy()

    sh = np.shape(A)
    nc = sh[0]
    nx = sh[1]
    
    if nx == 1:
        # Polytope is a 1-dim line
        for ii in xrange(nc):
            V = np.append(V, b[ii]/A[ii])
        if len(A) == 1:
            R = np.append(R,1)
            raise Exception("extreme: polytope is unbounded")
    
    elif nx == 2:
        # Polytope is 2D
        alf = np.angle(A[:,0]+1j*A[:,1])
        I = np.argsort(alf)
        #Y = alf[I]
        H = np.vstack([A, A[0,:]])
        K = np.hstack([b, b[0]])
        I = np.hstack([I,I[0]])
        for ii in xrange(nc):
            HH = np.vstack([H[I[ii],:],H[I[ii+1],:]])
            KK = np.hstack([K[I[ii]],K[I[ii+1]]])
            if np.linalg.cond(HH) == np.inf:
                R = np.append(R,1)
                raise Exception("extreme: polytope is unbounded")
            else:
                try:
                    v = np.linalg.solve(HH, KK)
                except:
                    msg = 'Finding extreme points failed, '
                    msg += 'Check if any unbounded Polytope '
                    msg += 'is causing this.'
                    raise Exception(msg)
                if len(V) == 0:
                    V = np.append(V,v)
                else:
                    V = np.vstack([V,v])    
    else:
        # General nD method,
        # solve a vertex enumeration problem for
        # the dual polytope
        xmid = poly1.x
        A = poly1.A.copy()
        b = poly1.b.copy()
        sh = np.shape(A)
        Ai = np.zeros(sh)
        
        for ii in xrange(sh[0]):
            Ai[ii,:] = A[ii,:]/(b[ii]-np.dot(A[ii,:],xmid))
        
        Q = reduce(qhull(Ai))
                
        if not Q.is_fulldim():
            return None
        
        H = Q.A
        K = Q.b
                
        sh = np.shape(H)
        nx = sh[1]
        V = np.zeros(sh)
        for iv in xrange(sh[0]):
            for ix in xrange(nx):
                V[iv,ix] = H[iv,ix]/K[iv] + xmid[ix]
    
    return V.reshape((V.size/nx, nx))

def qhull(vertices, abs_tol=ABS_TOL):
    """Return convex hull of C{vertices} computed by quickhull.
    
    @param vertices: N x d array containing N vertices
        in dimension d
    
    @return: convex hull
    @rtype: L{Polytope}
    """
    A, b, vert = quickhull(vertices, abs_tol=abs_tol)
    if A.size == 0:
        return Polytope()
    p = ConvexPolytope(A, b, minrep=True)
    p._vertices = vertices
    return Polytope([p])

def _projection(poly1, dim, solver=None, abs_tol=ABS_TOL, verbose=0):
    """Projects a polytope onto lower dimensions.
    
    Available solvers are:

      - "esp": Equality Set Projection;
      - "exthull": vertex projection;
      - "fm": Fourier-Motzkin projection;
      - "iterhull": iterative hull method.
    
    Example:
    To project the polytope `P` onto the first three dimensions, use
        >>> P_proj = projection(P, [1,2,3])

    @param poly1: Polytope to project
    @type poly1: L{ConvexPolytope}
    
    @param dim: Dimensions on which to project
    @param solver: A solver can be specified, if left blank an attempt
        is made to choose the most suitable solver.
    @param verbose: if positive, print solver used in case of
        guessing; default is 0 (be silent).

    @rtype: L{Polytope}
    @return: Projected polytope in lower dimension
    """
    if (poly1.dim < len(dim)) or poly1.is_empty():
        return poly1
    
    poly_dim = poly1.dim
    dim = np.array(dim)
    org_dim = xrange(poly_dim)
    new_dim = dim.flatten() - 1
    del_dim = np.setdiff1d(org_dim,new_dim) # Index of dimensions to remove 
    
    logger.debug('polytope dim = ' + str(poly_dim))
    logger.debug('project on dims = ' + str(new_dim))
    logger.debug('original dims = ' + str(org_dim))
    logger.debug('dims to delete = ' + str(del_dim))
    
    mA, nA = poly1.A.shape
    
    if mA < poly_dim:
        msg = 'fewer rows in A: ' + str(mA)
        msg += ', than polytope dimension: ' + str(poly_dim)
        logger.warn(msg)
        
        # enlarge A, b with zeros
        A = poly1.A.copy()
        poly1.A = np.zeros((poly_dim, poly_dim) )
        poly1.A[0:mA, 0:nA] = A
        
        poly1.b = np.hstack([poly1.b, np.zeros(poly_dim - mA)])
    
    logger.debug('m, n = ' +str((mA, nA)) )
    
    # Compute cheby ball in lower dim to see if projection exists
    norm = np.sum(poly1.A*poly1.A, axis=1).flatten()
    norm[del_dim] = 0
    c = matrix(np.zeros(len(org_dim)+1, dtype=float))
    c[len(org_dim)] = -1
    G = matrix(np.hstack([poly1.A, norm.reshape(norm.size,1)]))
    h = matrix(poly1.b)
    sol = solvers.lp(c,G,h,None,None,lp_solver)
    if sol['status'] != "optimal":
        # Projection not fulldim
        return ConvexPolytope()
    if sol['x'][-1] < abs_tol:
        return ConvexPolytope()
    
    if solver == "esp":
        return _projection_esp(poly1,new_dim, del_dim)
    elif solver == "exthull":
        return _projection_exthull(poly1,new_dim)
    elif solver == "fm":
        return _projection_fm(poly1,new_dim,del_dim)
    elif solver == "iterhull": 
        return _projection_iterhull(poly1,new_dim)
    elif solver is not None:
        logger.warn('unrecognized projection solver "' +
                    str(solver) + '".')
    
    if len(del_dim) <= 2:
        logger.debug("projection: using Fourier-Motzkin.")
        return _projection_fm(poly1,new_dim,del_dim)
    elif len(org_dim) <= 4:
        logger.debug("projection: using exthull.")
        return _projection_exthull(poly1,new_dim)
    else:
        logger.debug("projection: using iterative hull.")
        return _projection_iterhull(poly1,new_dim)
        
def separate(reg1, abs_tol=ABS_TOL):
    """Divide a region into several regions such that they are
    all connected.
    
    @type reg1: L{Polytope}
    @param abs_tol: Absolute tolerance
    
    @return: List of connected Polytopes
    """
    final = []
    ind_left = xrange(len(reg1))
    
    props = reg1.props
    
    while len(ind_left) > 0:
        ind_del = []
        connected_reg = Polytope(
            [reg1[ind_left[0]]],
            []
        )
        ind_del.append(ind_left[0])
        for i in xrange(1,len(ind_left)):
            j = ind_left[i]
            if connected_reg.is_adjacent(reg1[j]):
                connected_reg = connected_reg.union(
                    reg1[j],
                    check_convex = False
                )
                ind_del.append(j)
        
        connected_reg.props = props.copy()
        final.append(connected_reg)
        ind_left = np.setdiff1d(ind_left, ind_del)
    
    return final

def _is_adjacent(poly1, poly2, overlap=True, abs_tol=ABS_TOL):
    """Return True if convex polytopes are adjacent.
    
    Check by enlarging both slightly and checking for intersection.
    
    @type poly1: L{ConvexPolytope}
    @type poly2: L{ConvexPolytope}
    
    @param overlap: return True if polytopes are neighbors OR overlap
    
    @param abs_tol: absolute tolerance
    
    @return: True if polytopes are adjacent
    """
    if not isinstance(poly1, ConvexPolytope):
        raise TypeError('poly1 is not a ConvexPolytope.')
    
    if not isinstance(poly1, ConvexPolytope):
        raise TypeError('poly2 is not a ConvexPolytope.')
    
    if poly1.dim != poly2.dim:
        raise Exception("is_adjacent: "
            "polytopes do not have the same dimension")
        
    A1_arr = poly1.A.copy()
    A2_arr = poly2.A.copy()
    b1_arr = poly1.b.copy()
    b2_arr = poly2.b.copy()
    
    if overlap:
        b1_arr += abs_tol
        b2_arr += abs_tol 
        dummy = ConvexPolytope(
            np.concatenate((A1_arr, A2_arr)),
            np.concatenate((b1_arr, b2_arr))
        )
        return dummy.is_fulldim(abs_tol=abs_tol / 10)
        
    else:
        M1 = np.concatenate((poly1.A, np.array([poly1.b]).T), 1).T
        M1row = 1 / np.sqrt(np.sum(M1**2, 0))
        M1n = np.dot(M1, np.diag(M1row))
        
        M2 = np.concatenate((poly2.A, np.array([poly2.b]).T), 1).T
        M2row = 1 / np.sqrt(np.sum(M2**2, 0))
        M2n = np.dot(M2, np.diag(M2row))
        
        if not np.any(np.dot(M1n.T,M2n) < -0.99):
            return False      
        
        dummy = np.dot(M1n.T, M2n)
        row, col = np.nonzero(np.isclose(dummy, dummy.min() ) )
        
        for i,j in zip(row, col):
            b1_arr[i] += abs_tol
            b2_arr[j] += abs_tol
        
        dummy = ConvexPolytope(
            np.concatenate((A1_arr, A2_arr)),
            np.concatenate((b1_arr, b2_arr))
        )
        return dummy.is_fulldim(abs_tol=abs_tol / 10)
    
#### Helper functions ####
        
def _projection_fm(poly1, new_dim, del_dim, abs_tol=ABS_TOL):
    """Help function implementing Fourier Motzkin projection.
    Should work well for eliminating few dimensions.
    """
    # Remove last dim first to handle indices
    del_dim = -np.sort(-del_dim)
     
    if not poly1.minrep:
        poly1 = poly1.reduction()
        
    poly = poly1.copy()
    
    for i in del_dim:
        positive = np.nonzero(poly.A[:,i] > abs_tol)[0]
        negative = np.nonzero(poly.A[:,i] < abs_tol)[0]
        null = np.nonzero(np.abs(poly.A[:,i]) < abs_tol)[0]
                
        nr = len(null)+ len(positive)*len(negative)
        nc = np.shape(poly.A)[0]
        C = np.zeros([nr,nc])
        
        A = poly.A[:,i].copy()
        row = 0
        for j in positive:
            for k in negative:
                C[row,j] = -A[k]
                C[row,k] = A[j]
                row += 1
        for j in null:
            C[row,j] = 1
            row += 1
        keep_dim = np.setdiff1d(
            range(poly.A.shape[1]),
            np.array([i])
        )
        poly = ConvexPolytope(
            np.dot(C,poly.A)[:,keep_dim],
            np.dot(C,poly.b)
        )
        if not poly.is_fulldim():
            return ConvexPolytope()
        poly = poly.reduction()
        
    return poly
    
def _projection_exthull(poly1,new_dim):
    """Help function implementing vertex projection.
    Efficient in low dimensions.
    """
    vert = poly1.vertices
    if vert is None:
        # qhull failed
        return ConvexPolytope(fulldim=False, minrep=True)
    return qhull(vert[:,new_dim]).reduction()
    
def _projection_iterhull(poly1, new_dim, max_iter=1000,
                        verbose=0, abs_tol=ABS_TOL):
    """Helper function implementing the "iterative hull" method.
    Works best when projecting _to_ lower dimensions.
    """
    r = poly1.r
    org_dim = poly1.A.shape[1]
            
    logger.debug("Starting iterhull projection from dim " +
                 str(org_dim) + " to dim " + str(len(new_dim)) )
            
    if len(new_dim) == 1:
        f1 = np.zeros(poly1.A.shape[1])
        f1[new_dim] = 1
        sol = solvers.lp(
            matrix(f1), matrix(poly1.A), matrix(poly1.b),
            None, None, lp_solver
        )
        if sol['status'] == "optimal":
            vert1 = sol['x']
        sol = solvers.lp(
            matrix(-f1), matrix(poly1.A), matrix(poly1.b),
            None, None, lp_solver
        )
        if sol['status'] == "optimal":
            vert2 = sol['x']
        vert = np.vstack([vert1,vert2])
        return qhull(vert)
        
    else:
        OK = False
        cnt = 0
        Vert = None
        while not OK:
            #Maximizing in random directions
            #to find a starting simplex
            cnt += 1
            if cnt > max_iter:  
                raise Exception("iterative_hull: "
                    "could not find starting simplex")
            
            f1 = np.random.rand(len(new_dim)).flatten() - 0.5
            f = np.zeros(org_dim)
            f[new_dim]=f1
            sol = solvers.lp(
                matrix(-f), matrix(poly1.A), matrix(poly1.b),
                None, None, lp_solver
            )
            xopt = np.array(sol['x']).flatten()  
            if Vert is None:
                Vert = xopt.reshape(1,xopt.size)
            else:
                k = np.nonzero( Vert[:,new_dim[0]] == xopt[new_dim[0]] )[0]
                for j in new_dim[range(1,len(new_dim))]:
                    ii = np.nonzero(Vert[k,j] == xopt[j])[0]
                    k = k[ii]
                    if k.size == 0:
                        break
                if k.size == 0:
                    Vert = np.vstack([Vert,xopt])
            
            if Vert.shape[0] > len(new_dim):
                u, s, v = np.linalg.svd(
                    np.transpose(Vert[:,new_dim] - Vert[0,new_dim])
                )
                rank = np.sum(s > abs_tol*10)
                if rank == len(new_dim):
                    # If rank full we have found a starting simplex
                    OK = True
                    
        logger.debug("Found starting simplex after " +
                     str(cnt) +" iterations")
        
        cnt = 0
        P1 = qhull(Vert[:,new_dim])            
        HP = None
        
        while True:
            # Iteration:
            # Maximaze in direction of each facet
            # Take convex hull of all vertices
            cnt += 1     
            if cnt > max_iter:
                raise Exception("iterative_hull: "
                    "maximum number of iterations reached")
            
            logger.debug("Iteration number " + str(cnt) )
            
            for ind in xrange(P1.A.shape[0]):
                f1 = np.round(P1.A[ind,:]/abs_tol)*abs_tol
                f2 = np.hstack([np.round(P1.A[ind,:]/abs_tol)*abs_tol, \
                     np.round(P1.b[ind]/abs_tol)*abs_tol])
                                
                # See if already stored
                k = np.array([])
                if HP is not None:
                    k = np.nonzero( HP[:,0] == f2[0] )[0]
                    for j in xrange(1,np.shape(P1.A)[1]+1):
                        ii = np.nonzero(HP[k,j] == f2[j])[0]
                        k = k[ii]
                        if k.size == 0:
                            break
                
                if k.size == 1:
                    # Already stored
                    xopt = HP[
                        k,
                        range(
                            np.shape(P1.A)[1]+1,
                            np.shape(P1.A)[1] + np.shape(Vert)[1] + 1
                        )
                    ]
                else:
                    # Solving optimization to find new vertex
                    f = np.zeros(poly1.A.shape[1])
                    f[new_dim]=f1
                    sol = solvers.lp(
                        matrix(-f), matrix(poly1.A), matrix(poly1.b),
                        None, None, lp_solver
                    )
                    if sol['status'] != 'optimal':
                        logger.error("iterhull: LP failure")
                        continue
                    xopt = np.array(sol['x']).flatten()
                    add = np.hstack([f2, np.round(xopt/abs_tol)*abs_tol])
                    
                    # Add new half plane information
                    # HP format: [ P1.Ai P1.bi xopt]
                    if HP is None:
                        HP = add.reshape(1,add.size)
                    else:
                        HP = np.vstack([HP,add])
                        
                    Vert = np.vstack([Vert, xopt])
            
            logger.debug("Taking convex hull of new points")
            
            P2 = qhull(Vert[:,new_dim])
            
            logger.debug("Checking if new points are inside convex hull")
            
            OK = 1
            for i in xrange(np.shape(Vert)[0]):
                if not P1.__contains__(Vert[i,new_dim], abs_tol=1e-5):
                    # If all new points are inside
                    # old polytope -> Finished
                    OK = 0
                    break
            if OK == 1:
                logger.debug("Returning projection after " +
                             str(cnt) +" iterations\n")
                return P2
            else:
                # Iterate
                P1 = P2
                
def _projection_esp(poly1,keep_dim,del_dim):
    """Helper function implementing "Equality set projection".
    Very buggy.
    """
    C = poly1.A[:,keep_dim]
    D = poly1.A[:,del_dim]
    if not poly1.is_fulldim():
        return ConvexPolytope()
    G,g,E = esp(C,D,poly1.b)
    return ConvexPolytope(G,g)

def _region_diff(poly, reg, abs_tol=ABS_TOL, intersect_tol=ABS_TOL,
                 save=False):
    """Subtract a region from a polytope
    
    @param poly: polytope from which to subtract a region
    @param reg: region which should be subtracted
    @param abs_tol: absolute tolerance
    
    @return: polytope or region containing non-overlapping polytopes
    """
    if not isinstance(poly, ConvexPolytope):
        raise Exception('poly not a Polytope, but: ' +
                        str(type(poly) ) )
    poly = poly.copy()
    
    if isinstance(reg, ConvexPolytope):
        reg = Polytope([reg])
    
    if not isinstance(reg, Polytope):
        raise Exception('reg not a Region, but: ' +
                        str(type(reg) ) )
    
    #Pdummy = poly
    
    N = len(reg)
    
    if isinstance(reg, ConvexPolytope):
        # Hack if reg happens to be a polytope
        reg = Polytope([reg])
        N = 1
        
    if reg.is_empty():
        return poly

    if poly.is_empty():
        return ConvexPolytope()
    
    # Checking intersections to find Polytopes in Region
    # that intersect the Polytope
    Rc = np.zeros(N)
    for i, poly1 in enumerate(reg):
        A_dummy = np.vstack([poly.A, poly1.A])
        b_dummy = np.hstack([poly.b, poly1.b])
        dummy = ConvexPolytope(A_dummy, b_dummy)
        Rc[i] = dummy.r

    N = np.sum(Rc >= intersect_tol)    
    if N == 0:
        logger.debug('no ConvexPolytope in the Polytope ' +
                     'intersects the given ConvexPolytope')
        return poly
    
    # Sort radii
    Rc = -Rc
    ind = np.argsort(Rc)
    #val = Rc[ind]
    
    A = poly.A.copy()
    B = poly.b.copy()
    H = A.copy()
    K = B.copy()
    m = np.shape(A)[0]
    mi = np.zeros([N,1], dtype=int)
    
    # Finding constraints that are not in original polytope
    HK = np.hstack([H,np.array([K]).T])
    for ii in xrange(N): 
        i = ind[ii]
        if not reg[i].is_fulldim():
            continue
        Hni = reg[i].A.copy()
        Kni = reg[i].b.copy()   
        
        for j in xrange(np.shape(Hni)[0]):
            HKnij = np.hstack([Hni[j,:], Kni[j]])
            HK2 = np.tile(HKnij,[m,1])
            abs = np.abs(HK-HK2)
            
            if np.all(np.sum(abs,axis=1) >= abs_tol):
                # The constraint HKnij is not in original polytope
                mi[ii]=mi[ii]+1
                A = np.vstack([A, Hni[j,:]])
                B = np.hstack([B, Kni[j]])
                
        
    if np.any(mi == 0):
    # If some Ri has no active constraints, Ri covers R
        return ConvexPolytope()
        
    M = np.sum(mi)
    
    if len( mi[0:len(mi)-1]) > 0:
        csum = np.cumsum(np.vstack([0,mi[0:len(mi)-1]]))
        beg_mi = csum + m*np.ones(len(csum),dtype = int) 
    else:
        beg_mi = np.array([m])
    
    A = np.vstack([A, -A[range(m,m+M),:]])
    B = np.hstack([B, -B[range(m,m+M)]])

    counter = np.zeros([N,1], dtype=int)
    INDICES = np.arange(m, dtype=int)
        
    level = 0
    res_count = 0
    res = Polytope() # Initiate output
    while level != -1:
        if save:
            if res:
                ax = res.plot()
                ax.axis([0.0, 1.0, 0.0, 2.0])
                ax.figure.savefig('./img/res' + str(res_count) + '.pdf')
                res_count += 1
        
        if counter[level] == 0:
            if save:
                logger.debug('counter[level] is 0')
            
            for j in xrange(level,N):
                auxINDICES = np.hstack([
                    INDICES,
                    range(beg_mi[j],beg_mi[j]+mi[j])
                ])
                Adummy = A[auxINDICES,:]
                bdummy = B[auxINDICES]
                R = ConvexPolytope(Adummy,bdummy).r
                if R > abs_tol:
                    level = j
                    counter[level] = 1
                    INDICES = np.hstack([INDICES, beg_mi[level]+M])
                    break
            
            if R < abs_tol:
                level = level - 1
                res = res.union(ConvexPolytope(A[INDICES,:],B[INDICES]), False)
                nzcount = np.nonzero(counter)[0]
                for jj in xrange(len(nzcount)-1,-1,-1):

                    if counter[level] <= mi[level]:
                        INDICES[len(INDICES)-1] = INDICES[len(INDICES)-1] -M
                        INDICES = np.hstack([
                            INDICES,
                            beg_mi[level] + counter[level] + M
                        ])
                        break
                    else:
                        counter[level] = 0
                        INDICES = INDICES[0:m+sum(counter)]
                        if level == -1:
                            logger.debug('returning res from 1st point')
                            return res
        else:
            if save:
                logger.debug('counter[level] > 0')
            
            # counter(level) > 0
            nzcount = np.nonzero(counter)[0]
            
            for jj in xrange(len(nzcount)-1,-1,-1):
                level = nzcount[jj]
                counter[level] = counter[level] + 1

                if counter[level] <= mi[level]:
                    INDICES[len(INDICES)-1] = INDICES[len(INDICES)-1] - M
                    INDICES = np.hstack([
                        INDICES,
                        beg_mi[level]+counter[level]+M-1
                    ])
                    break
                else:
                    counter[level] = 0
                    INDICES = INDICES[0:m+np.sum(counter)]
                    level = level - 1
                    if level == -1:
                        if save:
                            if save:
                                if res:
                                    ax = res.plot()
                                    ax.axis([0.0, 1.0, 0.0, 2.0])
                                    ax.figure.savefig('./img/res_returned' + str(res_count) + '.pdf')
                            logger.debug('returning res from 2nd point')
                        return res
                    
        test_poly = ConvexPolytope(A[INDICES,:],B[INDICES])
        rc = test_poly.r
        if rc > abs_tol:
            if level == N - 1:
                res = res.union(Polytope([test_poly.reduction()]), False)
            else:
                level = level + 1
    logger.debug('returning res from end')
    return res
    
def _num_bin(N, places=8):
    """Return N as list of bits, zero-filled to places.

    E.g., given N=7, num_bin returns [1, 1, 1, 0, 0, 0, 0, 0].
    """
    return [(N>>k)&0x1  for k in xrange(places)]

def box2convex_poly(box):
    """Return new ConvexPolytope from box.
    
    @param box: defining the convex Polytope
    @type box: [[x1min, x1max], [x2min, x2max],...]
    """
    return ConvexPolytope.from_box(box)

def boxes2poly(boxes):
    """Return new Polytope from list of boxes.
    
    @param boxes: defines the Polytope
    @type boxes: [((x1min, x1max), (x2min, x2max),...),
                  ...]
    """
    polys = list()
    for box in boxes:
        p = box2convex_poly(box)
        polys.append(p)
    return Polytope(polys)

def _get_patch(poly1, **kwargs):
    """Return matplotlib patch for given Polytope.

    Example::

    > # Plot Polytope objects poly1 and poly2 in the same plot
    > import matplotlib.pyplot as plt
    > fig = plt.figure()
    > ax = fig.add_subplot(111)
    > p1 = _get_patch(poly1, color="blue")
    > p2 = _get_patch(poly2, color="yellow")
    > ax.add_patch(p1)
    > ax.add_patch(p2)
    > ax.set_xlim(xl, xu) # Optional: set axis max/min
    > ax.set_ylim(yl, yu) 
    > plt.show()

    @type poly1: L{Polytope}
    @param kwargs: any keyword arguments valid for
        matplotlib.patches.Polygon
    """
    if mpl is None:
        logger.warn('matplotlib not found, no plotting.')
        return
    
    V = poly1.vertices
    
    xc = poly1.x
    x = V[:,1] - xc[1]
    y = V[:,0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x/mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2*(y < 0)
    angle = angle*corr
    ind = np.argsort(angle) 

    patch = mpl.patches.Polygon(V[ind,:], True, **kwargs)
    patch.set_zorder(0)
    return patch

def grid_region(polyreg, res=None):
    """Grid within polytope or region.
    
    @type polyreg: L{ConvexPolytope} or L{Polytope}
    
    @param res: resolution of grid
    """
    bbox = polyreg.bounding_box
    bbox = np.hstack(bbox)
    dom = bbox.flatten()
    
    density = 8
    if not res:
        res = []
        for i in xrange(0, dom.size, 2):
            L = dom[i+1] -dom[i]
            res += [density *L]
    x = dom2vec(dom, res)
    x = x[:, polyreg.are_inside(x) ]
    
    return (x, res)

def _plot_text(polyreg, txt, ax, color):
    if newax is None:
        logger.warn('pyvectorized not found. No plotting.')
        return None
    
    if ax is None:
        ax, fig = newax()
    
    xc = polyreg.x
    ax.text(xc[0], xc[1], txt, color=color)

def simplices2polytopes(points, triangles):
    """Convert a simplicial mesh to polytope H-representation.
    
    @type points: N x d
    
    @type triangles: NT x 3
    """
    polytopes = []
    for triangle in triangles:
        logger.debug('Triangle: ' + str(triangle))
        
        triangle_vertices = points[triangle, :]
        
        logger.debug('\t triangle points: ' +
                    str(triangle_vertices))
        
        poly = qhull(triangle_vertices)
        
        logger.debug('\n Polytope:\n:' + str(poly))
        
        polytopes += [poly]
    return polytopes
