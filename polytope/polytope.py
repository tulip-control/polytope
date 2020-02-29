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
r"""Computational geometry module for polytope computations.

For linear programming the fastest installed solver is selected.
To change this choice, see the module `polytope.solvers`.

The structure of this module is based on \cite{MPT04}.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import warnings

import numpy as np

from polytope.solvers import lpsolve
from polytope.esp import esp
from polytope.quickhull import quickhull


logger = logging.getLogger(__name__)
try:
    xrange
except NameError:
    xrange = range

# Nicer numpy output
np.set_printoptions(precision=5, suppress=True)

# global default absolute tolerance,
# to enable changing it code w/o passing arguments,
# so that magic methods can still be used
ABS_TOL = 1e-7

# inline imports:
#
# import matplotlib as mpl
# from matplotlib import pyplot as plt


class Polytope(object):
    """Polytope class with following fields

      - `A`: a numpy array for the hyperplane normals in hyperplane
             representation of a polytope
      - `b`: a numpy array for the hyperplane offsets in hyperplane
             representation of a polytope
      - `chebXc`: coordinates of chebyshev center (if calculated)
      - `chebR`: chebyshev radius (if calculated)
      - `bbox`: bounding box (if calculated)
      - `minrep`: if polytope is in minimal representation (after
                  running reduce)
      - `normalize`: if True (default), normalize given A and b arrays;
                     else, use A and b without modification.
      - `dim`: dimension
      - `volume`: volume, computed on first call

    See Also
    ========
    L{Region}
    """

    def __init__(
            self, A=np.array([]), b=np.array([]), minrep=False,
            chebR=0, chebX=None, fulldim=None,
            volume=None, vertices=None, normalize=True):
        self.A = A.astype(float)
        self.b = b.astype(float).flatten()
        if A.size > 0 and normalize:
            # Normalize
            Anorm = np.sqrt(np.sum(A * A, 1)).flatten()
            pos = np.nonzero(Anorm > 1e-10)[0]
            self.A = self.A[pos, :]
            self.b = self.b[pos]
            Anorm = Anorm[pos]
            mult = 1 / Anorm
            for i in xrange(self.A.shape[0]):
                self.A[i, :] = self.A[i, :] * mult[i]
            self.b = self.b.flatten() * mult
        self.minrep = minrep
        self._chebXc = chebX
        self._chebR = chebR
        self.bbox = None
        self.fulldim = fulldim
        self._volume = volume
        self.vertices = vertices

    def __str__(self):
        """Return pretty-formatted H-representation of polytope."""
        A, b = self.A, self.b
        A_rows = str(A).split('\n')
        n_rows = len(A_rows)
        # column vector from `b`, if not already one
        b_col = b.reshape(b.shape[0], 1) if len(b.shape) == 1 else b
        b_rows = str(b_col).split('\n')
        # place an "x" somewhere near the middle
        x_row = int((n_rows - 1) / 2)  # where "x" is shown
        above = x_row
        below = (n_rows - x_row - 2)
        spacer = ' |    '
        last_middle = [spacer[1:]] if n_rows > 1 else []
        middle = (
            above * [spacer]
            + [' x <= ']
            + below * [spacer]
            + last_middle)
        assert len(middle) == n_rows, (middle, n_rows)
        # format lines
        lines = [A_rows[k] + middle[k] + b_rows[k]
                 for k in range(n_rows)]
        output = 'Single polytope \n  {lines}\n'.format(
            lines='\n  '.join(lines))
        return output

    def __len__(self):
        return 0

    def __copy__(self):
        A = self.A.copy()
        b = self.b.copy()
        P = Polytope(A, b)
        P._chebXc = self._chebXc
        P._chebR = self._chebR
        P.minrep = self.minrep
        P.bbox = self.bbox
        P.fulldim = self.fulldim
        return P

    def __contains__(self, point):
        """Return `True` if `self` contains `point`.

        Boundary points are included.

        @param point: column vector, e.g., as `numpy.ndarray`
        @rtype: bool

        For multiple points, see the method `self.contains`.
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        test = self.A.dot(point.flatten()) - self.b < ABS_TOL
        return np.all(test)

    def contains(self, points, abs_tol=ABS_TOL):
        """Return Boolean array of whether each point in `self`.

        Any point that satisfies all inequalities is
        contained in `self`. A tolerance is added, and
        strict inequality checked (<). Pass `abs_tol=0`
        to exclude the boundary.

        @param points: column vectors
        @rtype: bool, 1d array
        """
        test = self.A.dot(points) - self.b[:, np.newaxis] < abs_tol
        return np.all(test, axis=0)

    def __eq__(self, other):
        return self <= other and other <= self

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return is_subset(self, other)

    def __ge__(self, other):
        return is_subset(other, self)

    def __bool__(self):
        return bool(self.volume > 0)

    __nonzero__ = __bool__

    def union(self, other, check_convex=False):
        """Return union with Polytope or Region.

        For usage see function union.

        @type other: L{Polytope} or L{Region}
        @rtype: L{Region}
        """
        return union(self, other, check_convex)

    def diff(self, other):
        """Return set difference with Polytope or Region.

        @type other: L{Polytope} or L{Region}
        @rtype: L{Region}
        """
        return mldivide(self, other)

    def intersect(self, other, abs_tol=ABS_TOL):
        """Return intersection with Polytope or Region.

        @type other: L{Polytope}.
        @rtype: L{Polytope} or L{Region}
        """
        if isinstance(other, Region):
            return other.intersect(self)
        if not isinstance(other, Polytope):
            msg = 'Polytope intersection defined only'
            msg += ' with other Polytope. Got instead: '
            msg += str(type(other))
            raise Exception(msg)
        if (not is_fulldim(self)) or (not is_fulldim(other)):
            return Polytope()
        if self.dim != other.dim:
            raise Exception("polytopes have different dimension")
        iA = np.vstack([self.A, other.A])
        ib = np.hstack([self.b, other.b])

        return reduce(Polytope(iA, ib), abs_tol=abs_tol)

    def translation(self, d):
        """Returns a copy of C{self} translated by the vector C{d}.

        Consult L{polytope.polytope._translate} for implementation details.

        @type d: 1d array
        """
        newpoly = self.copy()
        _translate(newpoly, d)
        return newpoly

    def rotation(self, i=None, j=None, theta=None):
        """Returns a rotated copy of C{self}.

        Describe the plane of rotation and the angle of rotation (in radians)
        with i, j, and theta.

        i and j are the indices 0..N-1 of two of the identity basis
        vectors, and theta is the angle of rotation.

        Consult L{polytope.polytope._rotate} for more detail.

        @type i: int
        @type j: int
        @type theta: number
        """
        newpoly = self.copy()
        _rotate(newpoly, i=i, j=j, theta=theta)
        return newpoly

    def copy(self):
        """Return copy of this Polytope."""
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
            except Exception:
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
        if (intervals[:, 0] > intervals[:, 1]).any():
            msg = 'Polytope.from_box: '
            msg += 'Invalid interval in from_box method.\n'
            msg += 'First element of an interval must'
            msg += ' not be larger than the second.'
            raise Exception(msg)
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.hstack([intervals[:, 1], -intervals[:, 0]])

        return cls(A, b, minrep=True)

    def project(self, dim, solver=None,
                abs_tol=ABS_TOL, verbose=0):
        """Return Polytope projection on selected subspace.

        For usage details see function: L{projection}.
        """
        return projection(self, dim, solver, abs_tol, verbose)

    def scale(self, factor):
        """Multiply polytope by scalar factor.

        A x <= b, becomes: A x <= (factor * b)

        @type factor: float
        """
        self.b = factor * self.b

    @property
    def dim(self):
        """Return Polytope dimension."""
        try:
            return np.shape(self.A)[1]
        except Exception:
            return 0.0

    @property
    def volume(self):
        if self._volume is None:
            self._volume = volume(self)
        return self._volume

    @property
    def chebR(self):
        r, xc = cheby_ball(self)
        return self._chebR

    @property
    def chebXc(self):
        r, xc = cheby_ball(self)
        return self._chebXc

    @property
    def cheby(self):
        return cheby_ball(self)

    @property
    def bounding_box(self):
        """Wrapper of L{polytope.bounding_box}.

        Computes the bounding box on first call.
        """
        if self.bbox is None:
            self.bbox = bounding_box(self)
        return self.bbox

    def plot(self, ax=None, color=None, hatch=None, alpha=1.0, linestyle=None, linewidth=None, edgecolor=None):
        if self.dim != 2:
            raise Exception("Cannot plot polytopes of dimension larger than 2")
        
        # Setting default values for plotting
        linestyle = linestyle or "dashed"
        linewidth = linewidth or 3
        edgecolor = edgecolor or "black"
        
        ax = _newax(ax)
        if not is_fulldim(self):
            logger.error("Cannot plot empty polytope")
            return None
        if color is None:
            color = np.random.rand(3)
        poly = _get_patch(
            self, facecolor=color, hatch=hatch,
            alpha=alpha, linestyle=linestyle, linewidth=linewidth,
            edgecolor=edgecolor)
        ax.add_patch(poly)
        return ax

    def text(self, txt, ax=None, color='black'):
        """Plot text at chebyshev center."""
        _plot_text(self, txt, ax, color)


def _translate(polyreg, d):
    """Translate C{polyreg} by the vector C{d}. Modifies C{polyreg} in-place.

    @type d: 1d array
    """
    if isinstance(polyreg, Polytope):
        # Translate hyperplanes
        polyreg.b = polyreg.b + np.dot(polyreg.A, d)
    else:
        # Translate subregions
        for poly in polyreg.list_poly:
            _translate(poly, d)
    # Translate bbox and cheby
    if polyreg.bbox is not None:
        polyreg.bbox = (polyreg.bbox[0] + d,
                        polyreg.bbox[1] + d)
    if polyreg._chebXc is not None:
        polyreg._chebXc = polyreg._chebXc + d


def _rotate(polyreg, i=None, j=None, u=None, v=None, theta=None, R=None):
    """Rotate C{polyreg} in-place. Return the rotation matrix.

    There are two types of rotation: simple and compound. For simple rotations,
    by definition, all motion can be projected as circles in a single plane;
    the other N - 2 dimensions are invariant. Therefore any simple rotation can
    be parameterized by its plane of rotation. Compound rotations are the
    combination of multiple simple rotations; they have more than one plane of
    rotation. For N > 3 dimensions, a compound rotation may be necessary to map
    one orientation to another (Euler's rotation theorem no longer applies).

    Use one of the following three methods to specify rotation. The first two
    can only express simple rotation, but simple rotations may be applied in a
    sequence to achieve a compound rotation.

    (1) Provide the indices 0..N-1 of the identity basis vectors, i and j,
    which define the plane of rotation and a radian angle of rotation, theta,
    between them. This method contructs the Givens rotation matrix. The right
    hand rule defines the positive rotation direction.

    (2) Provide two vectors, the two vectors define the plane of rotation
    and angle of rotation is TWICE the angle from the first vector, u, to
    the second vector, v.

    (3) Provide an N-by-N rotation matrix, R. WARNING: No checks are made to
    determine whether the provided transformation matrix is a valid rotation.

    Further Reading
    https://en.wikipedia.org/wiki/Plane_of_rotation

    @param polyreg: The polytope or region to be rotated.
    @type polyreg: L{Polytope} or L{Region}
    @param i: The first index describing the plane of rotation.
    @type i: int
    @param j: The second index describing the plane of rotation.
    @type j: int
    @param u: The first vector describing the plane of rotation.
    @type u: 1d array
    @param u: The second vector describing the plane of rotation.
    @type v: 1d array.
    @param theta: The radian angle to rotate the polyreg in the plane defined
                  by i and j.
    @type theta: number
    @param R: A predefined rotation matrix.
    @type R: 2d array
    """
    # determine the rotation matrix based on inputs
    if R is not None:
        logger.debug("rotate: R=\n{}".format(R))
        assert i is None, i
        assert j is None, j
        assert theta is None, theta
        assert u is None, u
        assert v is None, v
    elif i is not None and j is not None and theta is not None:
        logger.info("rotate via indices and angle.")
        assert R is None, R
        assert u is None, u
        assert v is None, v
        if i == j:
            raise ValueError("Must provide two unique basis vectors.")
        R = givens_rotation_matrix(i, j, theta, polyreg.dim)
    elif u is not None and v is not None:
        logger.info("rotate via 2 vectors.")
        assert R is None, R
        assert i is None, i
        assert j is None, j
        assert theta is None, theta
        R = solve_rotation_ap(u, v)
    else:
        raise ValueError("R or (i and j and theta) or (u and v) "
                         "must be defined.")
    if isinstance(polyreg, Polytope):
        # Ensure that half space is normalized before rotation
        n, p = _hessian_normal(polyreg.A, polyreg.b)
        # Rotate the hyperplane normals
        polyreg.A = np.inner(n, R)
        polyreg.b = p
    else:
        # Rotate subregions
        for poly in polyreg.list_poly:
            _rotate(poly, None, None, R=R)
    # transform bbox and cheby
    if polyreg.bbox is not None:
        polyreg.bbox = (np.inner(polyreg.bbox[0].T, R).T,
                        np.inner(polyreg.bbox[1].T, R).T)
    if polyreg._chebXc is not None:
        polyreg._chebXc = np.inner(polyreg._chebXc, R)
    return R


def givens_rotation_matrix(i, j, theta, N):
    """Return the Givens rotation matrix for an N-dimensional space."""
    R = np.identity(N)
    c = np.cos(theta)
    s = np.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def solve_rotation_ap(u, v):
    r"""Return the rotation matrix for the rotation in the plane defined by the
    vectors u and v across TWICE the angle between u and v.

    This algorithm uses the Aguilera-Perez Algorithm \cite{Aguilera}
    to generate the rotation matrix. The algorithm works basically as follows:

    Starting with the Nth component of u, rotate u towards the (N-1)th
    component until the Nth component is zero. Continue until u is parallel to
    the 0th basis vector. Next do the same with v until it only has none zero
    components in the first two dimensions. The result will be something like
    this:

    [[u0,  0, 0 ... 0],
     [v0, v1, 0 ... 0]]

    Now it is trivial to align u with v. Apply the inverse rotations to return
    to the original orientation.

    NOTE: The precision of this method is limited by sin, cos, and arctan
    functions.
    """
    # TODO: Assert vectors are non-zero and non-parallel aka exterior
    # product is non-zero
    N = u.size  # the number of dimensions
    uv = np.stack([u, v], axis=1)  # the plane of rotation
    M = np.identity(N)  # stores the rotations for rorienting reference frame
    # ensure u has positive basis0 component
    if uv[0, 0] < 0:
        M[0, 0] = -1
        M[1, 1] = -1
        uv = M.dot(uv)
    # align uv plane with the basis01 plane and u with basis0.
    for c in range(0, 2):
        for r in range(N - 1, c, -1):
            if uv[r, c] != 0:  # skip rotations when theta will be zero
                theta = np.arctan2(uv[r, c], uv[r - 1, c])
                Mk = givens_rotation_matrix(r, r - 1, theta, N)
                uv = Mk.dot(uv)
                M = Mk.dot(M)
    # rotate u onto v
    theta = 2 * np.arctan2(uv[1, 1], uv[0, 1])
    logger.debug(
        "solve_rotation_ap: {d} degree rotation".format(
            d=180 * theta / np.pi))
    R = givens_rotation_matrix(0, 1, theta, N)
    # perform M rotations in reverse order
    M_inverse = M.T
    R = M_inverse.dot(R.dot(M))
    return R


def _hessian_normal(A, b):
    """Normalize half space representation according to hessian normal form."""
    L2 = np.reshape(np.linalg.norm(A, axis=1), (-1, 1))  # needs to be column
    if any(L2 == 0):
        raise ValueError('One of the rows of A is a zero vector.')
    n = A / L2  # hyperplane normals
    p = b / L2.flatten()  # hyperplane distances from origin
    return n, p


class Region(object):
    """Class for lists of convex polytopes

    Contains the following fields:

      - `list_poly`: list of Polytope objects
      - `props`: set of propositions inside region
      - `bbox`: if calculated, bounding box of region (see bounding_box)
      - `fulldim`: if calculated, boolean indicating whether region is
                   fully dimensional
      - `dim`: dimension
      - `volume`: volume of region, calculated on first call
      - `chebXc`: coordinates of maximum chebyshev center (if calculated)
      - `chebR`: maximum chebyshev radius (if calculated)

    See Also
    ========
    L{Polytope}
    """

    def __init__(self, list_poly=None, props=None):
        if list_poly is None:
            list_poly = []
        if props is None:
            props = set()
        if isinstance(list_poly, str):
            # Hack to be able to use the Region class also for discrete
            # problems.
            self.list_poly = list_poly
            self.props = set(props)
        else:
            if isinstance(list_poly, Region):
                dim = list_poly[0].dim
                for poly in list_poly:
                    if poly.dim != dim:
                        raise Exception("Region error:"
                                        " Polytopes must be of same dimension!")
            self.list_poly = list_poly[:]
            for poly in list_poly:
                if is_empty(poly):
                    self.list_poly.remove(poly)
            self.props = set(props)
            self.bbox = None
            self.fulldim = None
            self._volume = None
            self._chebXc = None
            self._chebR = None

    def __iter__(self):
        return iter(self.list_poly)

    def __getitem__(self, key):
        return self.list_poly[key]

    def __str__(self):
        output = ''
        for i in xrange(len(self.list_poly)):
            output += '\t Polytope number ' + str(i + 1) + ':\n'
            poly_str = str(self.list_poly[i])
            poly_str = poly_str.replace('\n', '\n\t\t')
            output += '\t ' + poly_str + '\n'
        output += '\n'
        return output

    def __len__(self):
        return len(self.list_poly)

    def __contains__(self, point):
        """Return `True` if `self` contains `point`.

        See `Polytope.__contains__`.
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        return any(point in u for u in self.list_poly)

    def contains(self, points, abs_tol=ABS_TOL):
        """Return Boolean array of whether each point in `self`.

        See `Polytope.contains`.
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        assert points.shape[0] == self.dim, 'points should be column vectors'
        contained = np.full(points.shape[1], False, dtype=bool)
        for poly in self.list_poly:
            contained = np.logical_or(
                poly.contains(points, abs_tol),
                contained)
        return contained

    def __eq__(self, other):
        return self <= other and other <= self

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return is_subset(self, other)

    def __ge__(self, other):
        return is_subset(other, self)

    def __add__(self, other):
        """Return union with Polytope or Region.

        Applies convex simplification if possible.
        To turn off this check,
        use Region.union

        @type other: L{Polytope} or L{Region}
        @rtype: L{Region}
        """
        return union(self, other, check_convex=True)

    def __bool__(self):
        return bool(self.volume > 0)

    __nonzero__ = __bool__

    def union(self, other, check_convex=False):
        """Return union with Polytope or Region.

        For usage see function union.

        @type other: L{Polytope} or L{Region}
        @rtype: L{Region}
        """
        return union(self, other, check_convex)

    def __sub__(self, other):
        """Return set difference with Polytope or Region.

        @type other: L{Polytope} or L{Region}
        @rtype: L{Region}
        """
        return mldivide(self, other)

    def diff(self, other):
        """Return set difference with Polytope or Region.

        @type other: L{Polytope} or L{Region}
        @rtype: L{Region}
        """
        return mldivide(self, other)

    def __and__(self, other):
        """Return intersection with Polytope or Region.

        Absolute tolerance 1e-7 used.
        To select the absolute tolerance use
        method Region.intersect

        @type other: L{Polytope} or L{Region}
        @rtype: L{Polytope} or L{Region}
        """
        return intersect(self, other)

    def intersect(self, other, abs_tol=ABS_TOL):
        """Return intersection with Polytope or Region.

        @type other: iterable container of L{Polytope}.
        @rtype: L{Region}
        """
        if isinstance(other, Polytope):
            other = [other]
        P = Region()
        for poly0 in self:
            for poly1 in other:
                isect = poly0.intersect(poly1, abs_tol)
                rp, xp = isect.cheby
                if rp > abs_tol:
                    P = union(P, isect, check_convex=True)
        return P

    def rotation(self, i=None, j=None, theta=None):
        """Returns a rotated copy of C{self}.

        Describe the plane of rotation and the angle of rotation (in radians)
        with i, j, and theta.

        i and j are the indices 0..N-1 of two of the identity basis
        vectors, and theta is the angle of rotation.

        Consult L{polytope.polytope._rotate} for more detail.

        @type i: int
        @type j: int
        @type theta: number
        """
        newreg = self.copy()
        _rotate(newreg, i=i, j=j, theta=theta)
        return newreg

    def translation(self, d):
        """Returns a copy of C{self} translated by the vector C{d}.

        Consult L{polytope.polytope._translate} for implementation details.

        @type d: 1d array
        """
        newreg = self.copy()
        _translate(newreg, d)
        return newreg

    def __copy__(self):
        """Return copy of this Region."""
        return Region(list_poly=self.list_poly[:],
                      props=self.props.copy())

    def copy(self):
        """Return copy of this Region."""
        return self.__copy__()

    @property
    def dim(self):
        """Return Region dimension."""
        return np.shape(self.list_poly[0].A)[1]

    @property
    def volume(self):
        if self._volume is None:
            self._volume = volume(self)
        return self._volume

    @property
    def chebR(self):
        r, xc = cheby_ball(self)
        return self._chebR

    @property
    def chebXc(self):
        r, xc = cheby_ball(self)
        return self._chebXc

    @property
    def cheby(self):
        return cheby_ball(self)

    @property
    def bounding_box(self):
        """Wrapper of polytope.bounding_box.

        Computes the bounding box on first call.
        """
        if self.bbox is None:
            self.bbox = bounding_box(self)
        return self.bbox

    def plot(self, ax=None, color=None, hatch=None, alpha=1.0, linestyle=None, linewidth=None, edgecolor=None):
        """Plot a `polytope` on axes `ax`."""
        # TODO optional arg for text label
        if self.dim != 2:
            raise Exception("Cannot plot region of dimension larger than 2")
        if not is_fulldim(self):
            logger.error("Cannot plot empty region")
            return None
        ax = _newax(ax)
        if color is None:
            color = np.random.rand(3)
        for poly2 in self.list_poly:
            # TODO hatched polytopes in same region
            poly2.plot(ax, color=color, hatch=hatch, alpha=alpha, linestyle=linestyle, linewidth=linewidth,
                       edgecolor=edgecolor)
        return ax

    def text(self, txt, ax=None, color='black'):
        """Plot text at chebyshev center."""
        _plot_text(self, txt, ax, color)


def is_empty(polyreg):
    """Check if the description of a polytope is empty

    @param polyreg: L{Polytope} or L{Region} instance

    @return: Boolean indicating whether polyreg is empty
    """
    n = len(polyreg)
    if len(polyreg) == 0:
        try:
            return len(polyreg.A) == 0
        except Exception:
            return True
    else:
        N = np.zeros(n, dtype=int)
        for i in xrange(n):
            N[i] = is_empty(polyreg.list_poly[i])
        if np.all(N):
            return True
        else:
            return False


def is_fulldim(polyreg, abs_tol=ABS_TOL):
    """Check if a polytope or region has inner points.

    @param polyreg: L{Polytope} or L{Region} instance

    @return: Boolean that is True if inner points found, False
        otherwise.
    """
    # logger.debug('is_fulldim')
    if polyreg.fulldim is not None:
        return polyreg.fulldim
    lenP = len(polyreg)
    if lenP == 0:
        rc, xc = cheby_ball(polyreg)
        status = rc > abs_tol
    else:
        status = np.zeros(lenP)
        for ii in xrange(lenP):
            rc, xc = cheby_ball(polyreg.list_poly[ii])
            status[ii] = rc > abs_tol
        status = np.sum(status)
        status = status > 0
    polyreg.fulldim = status
    return status


def is_convex(reg, abs_tol=ABS_TOL):
    """Check if a region is convex.

    @type reg: L{Region}

    @return: result,envelope: result indicating if convex. if found to
        be convex the envelope describing the convex polytope is
        returned.
    """
    if not is_fulldim(reg):
        return True
    if len(reg) == 0:
        return True
    outer = envelope(reg)
    if is_empty(outer):
        # Probably because input polytopes were so small and ugly..
        return False, None
    Pl, Pu = reg.bounding_box
    Ol, Ou = outer.bounding_box
    bboxP = np.hstack([Pl, Pu])
    bboxO = np.hstack([Ol, Ou])
    if (
            sum(abs(bboxP[:, 0] - bboxO[:, 0]) > abs_tol) > 0 or
            sum(abs(bboxP[:, 1] - bboxO[:, 1]) > abs_tol) > 0):
        return False, None
    if is_fulldim(outer.diff(reg)):
        return False, None
    else:
        return True, outer


def is_inside(polyreg, point, abs_tol=ABS_TOL):
    """Return `point in polyreg`.

    @type point: `collections.abc.Sequence` or `numpy.ndarray`
    @rtype: bool
    """
    warnings.warn(
        'Write `point in polyreg` instead of '
        'calling this function.',
        DeprecationWarning)
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    return polyreg.contains(point[:, np.newaxis], abs_tol)[0]


def is_subset(small, big, abs_tol=ABS_TOL):
    """Return True if small \subseteq big.

    @type small: L{Polytope} or L{Region}
    @type big:   L{Polytope} or L{Region}

    @rtype: bool
    """
    for x in [small, big]:
        if not isinstance(x, (Polytope, Region)):
            msg = 'Not a Polytope or Region, got instead:\n\t'
            msg += str(type(x))
            raise TypeError(msg)
    diff = small.diff(big)
    volume = diff.volume
    if volume < abs_tol:
        return True
    else:
        return False


def reduce(poly, nonEmptyBounded=1, abs_tol=ABS_TOL):
    """Remove redundant inequalities from the hyperplane representation.

    Uses the algorithm described at [1],
    by solving one LP for each facet.

    [1] http://www.ifor.math.ethz.ch/~fukuda/polyfaq/node24.html

    Warning:
      - nonEmptyBounded == 0 case is not tested much.

    @type poly: L{Polytope} or L{Region}

    @return: Reduced L{Polytope} or L{Region} object
    """
    if isinstance(poly, Region):
        lst = []
        for poly2 in poly.list_poly:
            red = reduce(poly2)
            if is_fulldim(red):
                lst.append(red)
        if len(lst) > 0:
            return Region(lst, poly.props)
        else:
            return Polytope()
    # is `poly` already in minimal representation ?
    if poly.minrep:
        return poly
    if not is_fulldim(poly):
        return Polytope()
    # `poly` isn't flat
    A_arr = poly.A
    b_arr = poly.b
    # Remove rows with b = inf
    keep_row = np.nonzero(poly.b != np.inf)
    A_arr = A_arr[keep_row]
    b_arr = b_arr[keep_row]
    neq = np.shape(A_arr)[0]
    # first eliminate the linearly dependent rows
    # corresponding to the same hyperplane
    M1 = np.hstack([A_arr, np.array([b_arr]).T]).T
    M1row = 1 / np.sqrt(np.sum(M1**2, 0))
    M1n = np.dot(M1, np.diag(M1row))
    M1n = M1n.T
    keep_row = []
    for i in xrange(neq):
        keep_i = 1
        for j in xrange(i + 1, neq):
            if np.dot(M1n[i].T, M1n[j]) > 1 - abs_tol:
                keep_i = 0
        if keep_i:
            keep_row.append(i)
    A_arr = A_arr[keep_row]
    b_arr = b_arr[keep_row]
    neq, nx = A_arr.shape
    if nonEmptyBounded:
        if neq <= nx + 1:
            return Polytope(A_arr, b_arr)
    # Now eliminate hyperplanes outside the bounding box
    if neq > 3 * nx:
        lb, ub = Polytope(A_arr, b_arr).bounding_box
        # cand = -(np.dot((A_arr>0)*A_arr,ub-lb)
        #-(b_arr-np.dot(A_arr,lb).T).T<-1e-4)
        cand = ~ (np.dot((A_arr > 0) * A_arr, ub - lb) -
                  (np.array([b_arr]).T - np.dot(A_arr, lb)) < -1e-4)
        A_arr = A_arr[cand.squeeze()]
        b_arr = b_arr[cand.squeeze()]
    neq, nx = A_arr.shape
    if nonEmptyBounded:
        if neq <= nx + 1:
            return Polytope(A_arr, b_arr)
    del keep_row[:]
    for k in xrange(A_arr.shape[0]):
        f = -A_arr[k, :]
        G = A_arr
        h = b_arr
        h[k] += 0.1
        sol = lpsolve(f, G, h)
        h[k] -= 0.1
        if sol['status'] == 0:
            obj = -sol['fun'] - h[k]
            if obj > abs_tol:
                keep_row.append(k)
        elif sol['status'] == 3:
            keep_row.append(k)
    polyOut = Polytope(A_arr[keep_row], b_arr[keep_row])
    polyOut.minrep = True
    return polyOut


def union(polyreg1, polyreg2, check_convex=False):
    """Compute the union of polytopes or regions

    @type polyreg1: L{Polytope} or L{Region}
    @type polyreg2: L{Polytope} or L{Region}
    @param check_convex: if True, look for convex unions and simplify

    @return: region of non-overlapping polytopes describing the union
    """
    # logger.debug('union')
    if is_empty(polyreg1):
        return polyreg2
    if is_empty(polyreg2):
        return polyreg1
    if check_convex:
        s1 = intersect(polyreg1, polyreg2)
        if is_fulldim(s1):
            s2 = polyreg2.diff(polyreg1)
            s3 = polyreg1.diff(polyreg2)
        else:
            s2 = polyreg1
            s3 = polyreg2
    else:
        s1 = polyreg1
        s2 = polyreg2
        s3 = None
    lst = []
    if len(s1) == 0:
        if not is_empty(s1):
            lst.append(s1)
    else:
        for poly in s1.list_poly:
            if not is_empty(poly):
                lst.append(poly)
    if len(s2) == 0:
        if not is_empty(s2):
            lst.append(s2)
    else:
        for poly in s2.list_poly:
            if not is_empty(poly):
                lst.append(poly)
    if s3 is not None:
        if len(s3) == 0:
            if not is_empty(s3):
                lst.append(s3)
        else:
            for poly in s3.list_poly:
                if not is_empty(poly):
                    lst.append(poly)
    if check_convex:
        final = []
        N = len(lst)
        if N > 1:
            # Check convexity for each pair of polytopes
            while N > 0:
                templist = [lst[0]]
                for ii in xrange(1, N):
                    templist.append(lst[ii])
                    is_conv, env = is_convex(Region(templist))
                    if not is_conv:
                        templist.remove(lst[ii])
                for poly in templist:
                    lst.remove(poly)
                cvxpoly = reduce(envelope(Region(templist)))
                if not is_empty(cvxpoly):
                    final.append(reduce(cvxpoly))
                N = len(lst)
        else:
            final = lst
        ret = Region(final)
    else:
        ret = Region(lst)
    return ret


def cheby_ball(poly1):
    """Calculate Chebyshev radius and center for a polytope.

    The Chebyshev radius is defined here as the radius of a maximal
    inscribed ball of the given polytope. The center of a maximal ball
    is also returned, but note that unlike the radius, it is not
    necessarily unique. If input is a region, then a largest Chebyshev
    ball is returned.

    N.B., this function will return whatever it finds in attributes
    chebR and chbXc if not None, without (re)computing the Chebyshev ball.

    Example (low dimension):

    r1,x1 = cheby_ball(P, [1]) calculates the center and half the
    length of the longest line segment along the first coordinate axis
    inside polytope P

    @type poly1: L{Polytope}

    @return: rc,xc: Chebyshev radius rc (float) and center xc (numpy array)
    """
    #logger.debug('cheby ball')
    if (poly1._chebXc is not None) and (poly1._chebR is not None):
        # In case chebyshev ball already calculated and stored
        return poly1._chebR, poly1._chebXc
    if isinstance(poly1, Region):
        maxr = 0
        maxx = None
        for poly in poly1.list_poly:
            rc, xc = cheby_ball(poly)
            if rc > maxr:
                maxr = rc
                maxx = xc
        poly1._chebXc = maxx
        poly1._chebR = maxr
        return maxr, maxx
    if is_empty(poly1):
        return 0, None
    # `poly1` is nonempty
    r = 0
    xc = None
    A = poly1.A
    c = np.negative(np.r_[np.zeros(np.shape(A)[1]), 1])
    norm2 = np.sqrt(np.sum(A * A, axis=1))
    G = np.c_[A, norm2]
    h = poly1.b
    sol = lpsolve(c, G, h)
    if sol['status'] == 0:
        r = sol['x'][-1]
        if r < 0:
            return 0, None
        xc = sol['x'][0:-1]
    else:
        # Polytope is empty
        poly1 = Polytope(fulldim=False)
        return 0, None
    poly1._chebXc = np.array(xc)
    poly1._chebR = np.double(r)
    return poly1._chebR, poly1._chebXc


def bounding_box(polyreg):
    """Return smallest hyperbox containing polytope or region.

    If polyreg.bbox is not None,
    then it is returned without update.

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
    if polyreg.bbox is not None:
        return polyreg.bbox
    # For regions, calculate recursively for each
    # convex polytope and take maximum
    if isinstance(polyreg, Region):
        lenP = len(polyreg)
        dimP = polyreg.dim
        alllower = np.zeros([lenP, dimP])
        allupper = np.zeros([lenP, dimP])
        # collect lower and upper bounds
        for ii in xrange(0, lenP):
            bbox = polyreg.list_poly[ii].bounding_box
            ll, uu = bbox
            alllower[ii, :] = ll.T
            allupper[ii, :] = uu.T
        l = np.zeros([dimP, 1])
        u = np.zeros([dimP, 1])
        # compute endpoints
        for ii in xrange(0, dimP):
            l[ii] = min(alllower[:, ii])
            u[ii] = max(allupper[:, ii])
        polyreg.bbox = l, u
        return l, u
    # For a single convex polytope, solve an optimization problem
    (m, n) = np.shape(polyreg.A)
    In = np.eye(n)
    l = np.zeros([n, 1])
    u = np.zeros([n, 1])
    # lower corner
    for i in xrange(0, n):
        c = np.array(In[:, i])
        G = polyreg.A
        h = polyreg.b
        sol = lpsolve(c, G, h)
        if sol['status'] == 0:
            x = sol['x']
            l[i] = x[i]
    # upper corner
    for i in xrange(0, n):
        c = np.negative(np.array(In[:, i]))
        G = polyreg.A
        h = polyreg.b
        sol = lpsolve(c, G, h)
        if sol['status'] == 0:
            x = sol['x']
            u[i] = x[i]
    polyreg.bbox = l, u
    return l, u


def envelope(reg, abs_tol=ABS_TOL):
    """Compute envelope of a region.

    The envelope is the polytope defined by all "outer" inequalities a
    x < b such that {x | a x < b} intersection P = P for all polytopes
    P in the region. In other words we want to find all "outer"
    equalities of the region.

    If envelope can't be computed an empty polytope is returned

    @type reg: L{Region}
    @param abs_tol: Absolute tolerance for calculations

    @return: Envelope of input
    """
    Ae = None
    be = None
    nP = len(reg.list_poly)
    for i in xrange(nP):
        poly1 = reg.list_poly[i]
        outer_i = np.ones(poly1.A.shape[0])
        for ii in xrange(poly1.A.shape[0]):
            if outer_i[ii] == 0:
                # If inequality already discarded
                continue
            for j in xrange(nP):
                # Check for each polytope
                # if it intersects with inequality ii
                if i == j:
                    continue
                poly2 = reg.list_poly[j]
                testA = np.vstack([poly2.A, -poly1.A[ii, :]])
                testb = np.hstack([poly2.b, -poly1.b[ii]])
                testP = Polytope(testA, testb)
                rc, xc = cheby_ball(testP)
                if rc > abs_tol:
                    # poly2 intersects with inequality ii -> this inequality
                    # can not be in envelope
                    outer_i[ii] = 0
        ind_i = np.nonzero(outer_i)[0]
        if Ae is None:
            Ae = poly1.A[ind_i, :]
            be = poly1.b[ind_i]
        else:
            Ae = np.vstack([Ae, poly1.A[ind_i, :]])
            be = np.hstack([be, poly1.b[ind_i]])
    ret = reduce(Polytope(Ae, be))
    if is_fulldim(ret):
        return ret
    else:
        return Polytope()


count = 0


def mldivide(a, b, save=False):
    """Return set difference a \ b.

    @param a: L{Polytope} or L{Region}
    @param b: L{Polytope} to subtract

    @return: L{Region} describing the set difference
    """
    if isinstance(b, Polytope):
        b = Region([b])
    if isinstance(a, Region):
        logger.debug('mldivide got Region as minuend')
        P = Region()
        for poly in a:
            #assert(not is_fulldim(P.intersect(poly) ) )
            Pdiff = poly
            for poly1 in b:
                Pdiff = mldivide(Pdiff, poly1, save=save)
            P = union(P, Pdiff, check_convex=True)
            if save:
                global count
                count = count + 1
                # dump plot of `Pdiff`
                ax = Pdiff.plot()
                ax.axis([0.0, 1.0, 0.0, 2.0])
                ax.figure.savefig('./img/Pdiff' + str(count) + '.pdf')
                # dump plot of `P`
                ax = P.plot()
                ax.axis([0.0, 1.0, 0.0, 2.0])
                ax.figure.savefig('./img/P' + str(count) + '.pdf')
    elif isinstance(a, Polytope):
        logger.debug('a is Polytope')
        P = region_diff(a, b)
    else:
        raise Exception('a neither Region nor Polytope')
    return P


def intersect(poly1, poly2, abs_tol=ABS_TOL):
    """Compute the intersection between two polytopes or regions

    @type poly1: L{Polytope} or L{Region}
    @type poly2: L{Polytope} or L{Region}

    @return: Intersection of poly1 and poly2 described by a polytope
    """
    # raise NotImplementedError('Being removed,
    # use {Polytope, Region}.intersect instead')
    if isinstance(poly1, Region):
        return poly1.intersect(poly2)
    if isinstance(poly2, Region):
        return poly2.intersect(poly1)
    if not isinstance(poly1, Polytope):
        msg = 'poly1 not Region nor Polytope.'
        msg += 'Got instead: ' + str(type(poly1))
        raise Exception(msg)
    return poly1.intersect(poly2, abs_tol)


def volume(polyreg):
    """Approximately compute the volume of a Polytope or Region.

    A randomized algorithm is used.

    @type polyreg: L{Polytope} or L{Region}

    @return: Volume of input
    """
    if not is_fulldim(polyreg):
        return 0.0
    try:
        if polyreg._volume is not None:
            return polyreg._volume
    except Exception:
        logger.debug('computing volume...')
    # `Region` ?
    if isinstance(polyreg, Region):
        tot_vol = 0.
        for i in xrange(len(polyreg)):
            tot_vol += volume(polyreg.list_poly[i])
        polyreg._volume = tot_vol
        return tot_vol
    # `polyreg` is a `Polytope`
    n = polyreg.A.shape[1]
    if n == 1:
        N = 50
    elif n == 2:
        N = 500
    elif n == 3:
        N = 3000
    else:
        N = 10000
    l_b, u_b = polyreg.bounding_box
    x = (np.tile(l_b, (1, N))
         + np.random.rand(n, N)
         * np.tile(u_b - l_b, (1, N)))
    aux = (np.dot(polyreg.A, x)
           - np.tile(np.array([polyreg.b]).T, (1, N)))
    aux = np.nonzero(np.all(aux < 0, 0))[0].shape[0]
    vol = np.prod(u_b - l_b) * aux / N
    polyreg._volume = vol
    return vol


def extreme(poly1):
    """Compute the extreme points of a _bounded_ polytope

    @param poly1: Polytope in dimension d

    @return: A (N x d) numpy array containing the N vertices of poly1
    """
    if poly1.vertices is not None:
        # In case vertices already stored
        return poly1.vertices
    V = np.array([])
    R = np.array([])
    if isinstance(poly1, Region):
        raise Exception("extreme: not executable for regions")
    # `poly1` is a `Polytope`
    poly1 = reduce(poly1)  # Need to have polytope non-redundant!
    if not is_fulldim(poly1):
        return None
    # `poly1` isn't flat
    A = poly1.A.copy()
    b = poly1.b.copy()
    sh = np.shape(A)
    nc = sh[0]
    nx = sh[1]
    # distinguish cases by dimension
    if nx == 1:
        # Polytope is a 1-dim line
        for ii in xrange(nc):
            V = np.append(V, b[ii] / A[ii])
        if len(A) == 1:
            R = np.append(R, 1)
            raise Exception("extreme: polytope is unbounded")
    elif nx == 2:
        # Polytope is 2D
        alf = np.angle(A[:, 0] + 1j * A[:, 1])
        I = np.argsort(alf)
        H = np.vstack([A, A[0, :]])
        K = np.hstack([b, b[0]])
        I = np.hstack([I, I[0]])
        for ii in xrange(nc):
            HH = np.vstack([H[I[ii], :], H[I[ii + 1], :]])
            KK = np.hstack([K[I[ii]], K[I[ii + 1]]])
            if np.linalg.cond(HH) == np.inf:
                R = np.append(R, 1)
                raise Exception("extreme: polytope is unbounded")
            else:
                try:
                    v = np.linalg.solve(HH, KK)
                except Exception:
                    msg = 'Finding extreme points failed, '
                    msg += 'Check if any unbounded Polytope '
                    msg += 'is causing this.'
                    raise Exception(msg)
                if len(V) == 0:
                    V = np.append(V, v)
                else:
                    V = np.vstack([V, v])
    else:
        # General nD method,
        # solve a vertex enumeration problem for
        # the dual polytope
        rmid, xmid = cheby_ball(poly1)
        A = poly1.A.copy()
        b = poly1.b.copy()
        sh = np.shape(A)
        Ai = np.zeros(sh)
        for ii in xrange(sh[0]):
            Ai[ii, :] = A[ii, :] / (b[ii] - np.dot(A[ii, :], xmid))
        Q = reduce(qhull(Ai))
        if not is_fulldim(Q):
            return None
        # `Q` isn't flat
        H = Q.A
        K = Q.b
        sh = np.shape(H)
        nx = sh[1]
        V = np.zeros(sh)
        for iv in xrange(sh[0]):
            for ix in xrange(nx):
                V[iv, ix] = H[iv, ix] / K[iv] + xmid[ix]
    a = V.size / nx
    assert a.is_integer(), a
    a = int(a)
    poly1.vertices = V.reshape((a, nx))
    return poly1.vertices


def qhull(vertices, abs_tol=ABS_TOL):
    """Use quickhull to compute a convex hull.

    @param vertices: A N x d array containing N vertices in dimension d

    @return: L{Polytope} describing the convex hull
    """
    A, b, vert = quickhull(vertices, abs_tol=abs_tol)
    if A.size == 0:
        return Polytope()
    return Polytope(A, b, minrep=True, vertices=vert)


def projection(poly1, dim, solver=None, abs_tol=ABS_TOL, verbose=0):
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
    @param dim: Dimensions on which to project
    @param solver: A solver can be specified, if left blank an attempt
        is made to choose the most suitable solver.
    @param verbose: if positive, print solver used in case of
        guessing; default is 0 (be silent).

    @rtype: L{Polytope}
    @return: Projected polytope in lower dimension
    """
    if isinstance(poly1, Region):
        ret = Polytope()
        for i in xrange(len(poly1.list_poly)):
            p = projection(
                poly1.list_poly[i], dim,
                solver=solver, abs_tol=abs_tol)
            ret = ret + p
        return ret
    # flat ?
    if (poly1.dim < len(dim)) or is_empty(poly1):
        return poly1
    # `poly1` isn't flat
    poly_dim = poly1.dim
    dim = np.array(dim)
    org_dim = xrange(poly_dim)
    new_dim = dim.flatten() - 1
    del_dim = np.setdiff1d(org_dim, new_dim)  # Index of dimensions to remove
    # logging
    logger.debug('polytope dim = ' + str(poly_dim))
    logger.debug('project on dims = ' + str(new_dim))
    logger.debug('original dims = ' + str(org_dim))
    logger.debug('dims to delete = ' + str(del_dim))
    mA, nA = poly1.A.shape
    # fewer rows than dimensions ?
    if mA < poly_dim:
        msg = 'fewer rows in A: ' + str(mA)
        msg += ', than polytope dimension: ' + str(poly_dim)
        logger.warning(msg)
        # enlarge A, b with zeros
        A = poly1.A.copy()
        poly1.A = np.zeros((poly_dim, poly_dim))
        poly1.A[0:mA, 0:nA] = A
        # stack
        poly1.b = np.hstack([poly1.b, np.zeros(poly_dim - mA)])
    logger.debug('m, n = ' + str((mA, nA)))
    # Compute cheby ball in lower dim to see if projection exists
    norm = np.sum(poly1.A * poly1.A, axis=1).flatten()
    norm[del_dim] = 0
    c = np.zeros(len(org_dim) + 1, dtype=float)
    c[len(org_dim)] = -1
    G = np.hstack([poly1.A, norm.reshape(norm.size, 1)])
    h = poly1.b
    sol = lpsolve(c, G, h)
    if sol['status'] != 0:
        # Projection not fulldim
        return Polytope()
    if sol['x'][-1] < abs_tol:
        return Polytope()
    # select projection solver
    if solver == "esp":
        return projection_esp(poly1, new_dim, del_dim)
    elif solver == "exthull":
        return projection_exthull(poly1, new_dim)
    elif solver == "fm":
        return projection_fm(poly1, new_dim, del_dim)
    elif solver == "iterhull":
        return projection_iterhull(poly1, new_dim)
    elif solver is not None:
        logger.warning('unrecognized projection solver "' +
                       str(solver) + '".')
    # `solver` undefined or unknown
    # select method based on dimension criteria
    if len(del_dim) <= 2:
        logger.debug("projection: using Fourier-Motzkin.")
        return projection_fm(poly1, new_dim, del_dim)
    elif len(org_dim) <= 4:
        logger.debug("projection: using exthull.")
        return projection_exthull(poly1, new_dim)
    else:
        logger.debug("projection: using iterative hull.")
        return projection_iterhull(poly1, new_dim)


def separate(reg1, abs_tol=ABS_TOL):
    """Divide a region into several regions such that they are
    all connected.

    @type reg1: L{Region}
    @param abs_tol: Absolute tolerance

    @return: List [] of connected Regions
    """
    final = []
    ind_left = xrange(len(reg1))
    props = reg1.props
    while len(ind_left) > 0:
        ind_del = []
        connected_reg = Region(
            [reg1.list_poly[ind_left[0]]],
            [])
        ind_del.append(ind_left[0])
        for i in xrange(1, len(ind_left)):
            j = ind_left[i]
            if is_adjacent(connected_reg, reg1.list_poly[j]):
                connected_reg = union(
                    connected_reg,
                    reg1.list_poly[j],
                    check_convex=False)
                ind_del.append(j)
        connected_reg.props = props.copy()
        final.append(connected_reg)
        ind_left = np.setdiff1d(ind_left, ind_del)
    return final


def is_adjacent(poly1, poly2, overlap=True, abs_tol=ABS_TOL):
    """Return True if two polytopes or regions are adjacent.

    Check by enlarging both slightly and checking for intersection.

    @type poly1, poly2: L{Polytope}s or L{Region}s

    @param overlap: return True if polytopes are neighbors OR overlap

    @param abs_tol: absolute tolerance

    @return: True if polytopes are adjacent
    """
    if poly1.dim != poly2.dim:
        raise Exception("is_adjacent: "
                        "polytopes do not have the same dimension")
    if isinstance(poly1, Region):
        for p in poly1:
            adj = is_adjacent(p, poly2, overlap=overlap, abs_tol=abs_tol)
            if adj:
                return True
        return False
    if isinstance(poly2, Region):
        for p in poly2:
            adj = is_adjacent(poly1, p, overlap=overlap, abs_tol=abs_tol)
            if adj:
                return True
        return False
    # copy
    A1_arr = poly1.A.copy()
    A2_arr = poly2.A.copy()
    b1_arr = poly1.b.copy()
    b2_arr = poly2.b.copy()
    if overlap:
        b1_arr += abs_tol
        b2_arr += abs_tol
        dummy = Polytope(
            np.concatenate((A1_arr, A2_arr)),
            np.concatenate((b1_arr, b2_arr)))
        return is_fulldim(dummy, abs_tol=abs_tol / 10)
    else:
        M1 = np.concatenate((poly1.A, np.array([poly1.b]).T), 1).T
        M1row = 1 / np.sqrt(np.sum(M1**2, 0))
        M1n = np.dot(M1, np.diag(M1row))

        M2 = np.concatenate((poly2.A, np.array([poly2.b]).T), 1).T
        M2row = 1 / np.sqrt(np.sum(M2**2, 0))
        M2n = np.dot(M2, np.diag(M2row))
        if not np.any(np.dot(M1n.T, M2n) < -0.99):
            return False
        dummy = np.dot(M1n.T, M2n)
        row, col = np.nonzero(np.isclose(dummy, dummy.min()))
        for i, j in zip(row, col):
            b1_arr[i] += abs_tol
            b2_arr[j] += abs_tol
        dummy = Polytope(
            np.concatenate((A1_arr, A2_arr)),
            np.concatenate((b1_arr, b2_arr)))
        return is_fulldim(dummy, abs_tol=abs_tol / 10)


def is_interior(r0, r1, abs_tol=ABS_TOL):
    """Return True if r1 is strictly in the interior of r0.

    Checks if r1 enlarged by abs_tol
    is a subset of r0.

    @type r0: L{Polytope} or L{Region}
    @type r1: L{Polytope} or L{Region}

    @rtype: bool
    """
    if isinstance(r0, Polytope):
        r0 = Region([r0])
    if isinstance(r1, Polytope):
        r1 = Region([r1])
    for p in r1:
        A = p.A.copy()
        b = p.b.copy() + abs_tol
        dummy = Polytope(A, b)
        if not dummy <= r0:
            return True
    return False


#### Helper functions ####

def projection_fm(poly1, new_dim, del_dim, abs_tol=ABS_TOL):
    """Help function implementing Fourier Motzkin projection.

    Should work well for eliminating few dimensions.
    """
    # Remove last dim first to handle indices
    del_dim = -np.sort(-del_dim)
    if not poly1.minrep:
        poly1 = reduce(poly1)
    poly = poly1.copy()
    for i in del_dim:
        positive = np.nonzero(poly.A[:, i] > abs_tol)[0]
        negative = np.nonzero(poly.A[:, i] < abs_tol)[0]
        null = np.nonzero(np.abs(poly.A[:, i]) < abs_tol)[0]
        nr = len(null) + len(positive) * len(negative)
        nc = np.shape(poly.A)[0]
        C = np.zeros([nr, nc])
        A = poly.A[:, i].copy()
        row = 0
        for j in positive:
            for k in negative:
                C[row, j] = -A[k]
                C[row, k] = A[j]
                row += 1
        for j in null:
            C[row, j] = 1
            row += 1
        keep_dim = np.setdiff1d(
            range(poly.A.shape[1]),
            np.array([i]))
        poly = Polytope(
            np.dot(C, poly.A)[:, keep_dim],
            np.dot(C, poly.b))
        if not is_fulldim(poly):
            return Polytope()
        poly = reduce(poly)
    return poly


def projection_exthull(poly1, new_dim):
    """Help function implementing vertex projection.

    Efficient in low dimensions.
    """
    vert = extreme(poly1)
    if vert is None:
        # qhull failed
        return Polytope(fulldim=False, minrep=True)
    return reduce(qhull(vert[:, new_dim]))


def projection_iterhull(poly1, new_dim, max_iter=1000,
                        verbose=0, abs_tol=ABS_TOL):
    """Helper function implementing the "iterative hull" method.

    Works best when projecting _to_ lower dimensions.
    """
    r, xc = cheby_ball(poly1)
    org_dim = poly1.A.shape[1]
    logger.debug("Starting iterhull projection from dim " +
                 str(org_dim) + " to dim " + str(len(new_dim)))
    if len(new_dim) == 1:
        f1 = np.zeros(poly1.A.shape[1])
        f1[new_dim] = 1
        sol = lpsolve(f1, poly1.A, poly1.b)
        if sol['status'] == 0:
            vert1 = sol['x']
        sol = lpsolve(np.negative(f1), poly1.A, poly1.b)
        if sol['status'] == 0:
            vert2 = sol['x']
        vert = np.vstack([vert1, vert2])
        return qhull(vert)
    else:
        OK = False
        cnt = 0
        Vert = None
        while not OK:
            # Maximizing in random directions
            # to find a starting simplex
            cnt += 1
            if cnt > max_iter:
                raise Exception("iterative_hull: "
                                "could not find starting simplex")
            f1 = np.random.rand(len(new_dim)).flatten() - 0.5
            f = np.zeros(org_dim)
            f[new_dim] = f1
            sol = lpsolve(np.negative(f), poly1.A, poly1.b)
            xopt = np.array(sol['x']).flatten()
            if Vert is None:
                Vert = xopt.reshape(1, xopt.size)
            else:
                k = np.nonzero(Vert[:, new_dim[0]] == xopt[new_dim[0]])[0]
                for j in new_dim[range(1, len(new_dim))]:
                    ii = np.nonzero(Vert[k, j] == xopt[j])[0]
                    k = k[ii]
                    if k.size == 0:
                        break
                if k.size == 0:
                    Vert = np.vstack([Vert, xopt])
            if Vert.shape[0] > len(new_dim):
                u, s, v = np.linalg.svd(
                    np.transpose(Vert[:, new_dim] - Vert[0, new_dim]))
                rank = np.sum(s > abs_tol * 10)
                if rank == len(new_dim):
                    # If rank full we have found a starting simplex
                    OK = True
        logger.debug("Found starting simplex after " +
                     str(cnt) + " iterations")
        cnt = 0
        P1 = qhull(Vert[:, new_dim])
        HP = None
        while True:
            # Iteration:
            # Maximaze in direction of each facet
            # Take convex hull of all vertices
            cnt += 1
            if cnt > max_iter:
                raise Exception("iterative_hull: "
                                "maximum number of iterations reached")
            logger.debug("Iteration number " + str(cnt))
            for ind in xrange(P1.A.shape[0]):
                f1 = np.round(P1.A[ind, :] / abs_tol) * abs_tol
                f2 = np.hstack([np.round(P1.A[ind, :] / abs_tol) * abs_tol,
                                np.round(P1.b[ind] / abs_tol) * abs_tol])
                # See if already stored
                k = np.array([])
                if HP is not None:
                    k = np.nonzero(HP[:, 0] == f2[0])[0]
                    for j in xrange(1, np.shape(P1.A)[1] + 1):
                        ii = np.nonzero(HP[k, j] == f2[j])[0]
                        k = k[ii]
                        if k.size == 0:
                            break
                if k.size == 1:
                    # Already stored
                    xopt = HP[
                        k,
                        range(
                            np.shape(P1.A)[1] + 1,
                            np.shape(P1.A)[1] + np.shape(Vert)[1] + 1)
                    ]
                else:
                    # Solving optimization to find new vertex
                    f = np.zeros(poly1.A.shape[1])
                    f[new_dim] = f1
                    sol = lpsolve(np.negative(f), poly1.A, poly1.b)
                    if sol['status'] != 0:
                        logger.error("iterhull: LP failure")
                        continue
                    xopt = np.array(sol['x']).flatten()
                    add = np.hstack([f2, np.round(xopt / abs_tol) * abs_tol])
                    # Add new half plane information
                    # HP format: [ P1.Ai P1.bi xopt]
                    if HP is None:
                        HP = add.reshape(1, add.size)
                    else:
                        HP = np.vstack([HP, add])
                    Vert = np.vstack([Vert, xopt])
            logger.debug("Taking convex hull of new points")
            P2 = qhull(Vert[:, new_dim])
            logger.debug("Checking if new points are inside convex hull")
            OK = 1
            for i in xrange(np.shape(Vert)[0]):
                if not P1.contains(np.transpose([Vert[i, new_dim]]),
                                   abs_tol=1e-5):
                    # If all new points are inside
                    # old polytope -> Finished
                    OK = 0
                    break
            if OK == 1:
                logger.debug("Returning projection after " +
                             str(cnt) + " iterations\n")
                return P2
            else:
                # Iterate
                P1 = P2


def projection_esp(poly1, keep_dim, del_dim):
    """Helper function implementing "Equality set projection".

    CAUTION: Very buggy.
    """
    C = poly1.A[:, keep_dim]
    D = poly1.A[:, del_dim]
    if not is_fulldim(poly1):
        return Polytope()
    G, g, E = esp(C, D, poly1.b)
    return Polytope(G, g)


def region_diff(poly, reg, abs_tol=ABS_TOL, intersect_tol=ABS_TOL,
                save=False):
    """Subtract a region from a polytope

    @param poly: polytope from which to subtract a region
    @param reg: region which should be subtracted
    @param abs_tol: absolute tolerance

    @return: polytope or region containing non-overlapping polytopes
    """
    if not isinstance(poly, Polytope):
        raise Exception('poly not a Polytope, but: ' +
                        str(type(poly)))
    poly = poly.copy()
    if isinstance(reg, Polytope):
        reg = Region([reg])
    if not isinstance(reg, Region):
        raise Exception('reg not a Region, but: '
                        + str(type(reg)))
    N = len(reg)
    if N == 0:
        # Hack if reg happens to be a polytope
        reg = Region([reg])
        N = 1
    if is_empty(reg):
        return poly
    if is_empty(poly):
        return Polytope()
    # Checking intersections to find Polytopes in Region
    # that intersect the Polytope
    Rc = np.zeros(N)
    for i, poly1 in enumerate(reg):
        A_dummy = np.vstack([poly.A, poly1.A])
        b_dummy = np.hstack([poly.b, poly1.b])
        dummy = Polytope(A_dummy, b_dummy)
        Rc[i], xc = cheby_ball(dummy)
    N = np.sum(Rc >= intersect_tol)
    if N == 0:
        logger.debug('no Polytope in the Region intersects the given Polytope')
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
    mi = np.zeros(N, dtype=int)
    # Finding constraints that are not in original polytope
    HK = np.hstack([H, np.array([K]).T])
    for ii in xrange(N):
        i = ind[ii]
        if not is_fulldim(reg.list_poly[i]):
            continue
        Hni = reg.list_poly[i].A.copy()
        Kni = reg.list_poly[i].b.copy()
        for j in xrange(np.shape(Hni)[0]):
            HKnij = np.hstack([Hni[j, :], Kni[j]])
            HK2 = np.tile(HKnij, [m, 1])
            abs = np.abs(HK - HK2)
            # is the constraint `HKnij` not in the original polytope ?
            if np.all(np.sum(abs, axis=1) >= abs_tol):
                mi[ii] = mi[ii] + 1
                A = np.vstack([A, Hni[j, :]])
                B = np.hstack([B, Kni[j]])
    # If some Ri has no active constraints, Ri covers R
    if np.any(mi == 0):
        return Polytope()
    # some constraints are active
    M = np.sum(mi)
    if len(mi[0:len(mi) - 1]) > 0:
        csum = np.cumsum(np.hstack([0, mi[0:len(mi) - 1]]))
        beg_mi = csum + m * np.ones(len(csum), dtype=int)
    else:
        beg_mi = np.array([m])
    A = np.vstack([A, -A[range(m, m + M), :]])
    B = np.hstack([B, -B[range(m, m + M)]])
    counter = np.zeros([N, 1], dtype=int)
    INDICES = np.arange(m, dtype=int)
    level = 0
    res_count = 0
    res = Polytope()  # Initialize output
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

            for j in xrange(level, N):
                auxINDICES = np.hstack([
                    INDICES,
                    range(beg_mi[j], beg_mi[j] + mi[j])
                ])
                Adummy = A[auxINDICES, :]
                bdummy = B[auxINDICES]
                R, xopt = cheby_ball(Polytope(Adummy, bdummy))
                if R > abs_tol:
                    level = j
                    counter[level] = 1
                    INDICES = np.hstack([INDICES, beg_mi[level] + M])
                    break
            if R < abs_tol:
                level = level - 1
                res = union(res, Polytope(A[INDICES, :], B[INDICES]), False)
                nzcount = np.nonzero(counter)[0]
                for jj in xrange(len(nzcount) - 1, -1, -1):
                    if counter[level] <= mi[level]:
                        INDICES[len(INDICES) -
                                1] = INDICES[len(INDICES) - 1] - M
                        INDICES = np.hstack([
                            INDICES,
                            beg_mi[level] + counter[level] + M
                        ])
                        break
                    else:
                        counter[level] = 0
                        INDICES = INDICES[0:m + sum(counter)]
                        if level == -1:
                            logger.debug('returning res from 1st point')
                            return res
        else:
            if save:
                logger.debug('counter[level] > 0')
            # counter(level) > 0
            nzcount = np.nonzero(counter)[0]
            for jj in xrange(len(nzcount) - 1, -1, -1):
                level = nzcount[jj]
                counter[level] = counter[level] + 1
                if counter[level] <= mi[level]:
                    INDICES[len(INDICES) - 1] = INDICES[len(INDICES) - 1] - M
                    INDICES = np.hstack([
                        INDICES,
                        beg_mi[level] + counter[level] + M - 1
                    ])
                    break
                else:
                    counter[level] = 0
                    INDICES = INDICES[0:m + np.sum(counter)]
                    level = level - 1
                    if level == -1:
                        if save:
                            if save:
                                if res:
                                    ax = res.plot()
                                    ax.axis([0.0, 1.0, 0.0, 2.0])
                                    ax.figure.savefig('./img/res_returned'
                                                      + str(res_count)
                                                      + '.pdf')
                            logger.debug('returning res from 2nd point')
                        return res
        test_poly = Polytope(A[INDICES, :], B[INDICES])
        rc, xc = cheby_ball(test_poly)
        if rc > abs_tol:
            if level == N - 1:
                res = union(res, reduce(test_poly), False)
            else:
                level = level + 1
    logger.debug('returning res from end')
    return res


def num_bin(N, places=8):
    """Return N as list of bits, zero-filled to places.

    E.g., given N=7, num_bin returns [1, 1, 1, 0, 0, 0, 0, 0].
    """
    return [(N >> k) & 0x1 for k in xrange(places)]


def box2poly(box):
    """Return new Polytope from box.

    @param box: defining the Polytope
    @type box: [[x1min, x1max], [x2min, x2max],...]
    """
    return Polytope.from_box(box)


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
    import matplotlib as mpl
    V = extreme(poly1)
    rc, xc = cheby_ball(poly1)
    x = V[:, 1] - xc[1]
    y = V[:, 0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x / mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2 * (y < 0)
    angle = angle * corr
    ind = np.argsort(angle)
    # create patch
    patch = mpl.patches.Polygon(V[ind, :], True, **kwargs)
    patch.set_zorder(0)
    return patch


def grid_region(polyreg, res=None):
    """Return bounding box grid points within `polyreg`.

    @type polyreg: L{Polytope} or L{Region}
    @param res: resolution of grid
    """
    # grid corners
    bbox = polyreg.bounding_box
    bbox = np.hstack(bbox)
    dom = bbox.flatten()
    # grid resolution
    density = 8
    if res is None:
        res = list()
        for i in xrange(0, dom.size, 2):
            L = dom[i + 1] - dom[i]
            res += [density * L]
    linspaces = list()
    for i, n in enumerate(res):
        a = dom[2 * i]
        b = dom[2 * i + 1]
        r = np.linspace(a, b, n)
        linspaces.append(r)
    points = np.meshgrid(*linspaces)
    x = np.vstack(map(np.ravel, points))
    x = x[:, polyreg.contains(x)]
    return (x, res)


def _plot_text(polyreg, txt, ax, color):
    """Annotate center of Chebyshev ball with `txt`."""
    ax = _newax(ax)
    rc, xc = cheby_ball(polyreg)
    ax.text(xc[0], xc[1], txt, color=color)


def _newax(ax=None):
    """Add subplot to current figure and return axes."""
    from matplotlib import pyplot as plt
    if ax is not None:
        return ax
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return ax


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
