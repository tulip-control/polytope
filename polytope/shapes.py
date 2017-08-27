"""Class interfaces for geometric shapes."""
# Copyright (c) 2011-2017 by California Institute of Technology
# All rights reserved. Licensed under 3-clause BSD.
#
from __future__ import absolute_import
import logging


# It seems OK to implement some methods that don't preserve type,
# i.e., that return a `Polytope` but are methods of `Hull`.
# This is more practical.
#
# With the class naming in the new API, the users should create only
# `Polytopes`, so they should be mostly oblivious to `Hull`s,
# unless they want to implement some geometric algorithm themselves.
# Much like the difference between high-level and low-level BDD managers.


class Hull(object):
    """Convex hull of points.

    Another name is *convex polytope*.
    """

    def __init__(self):
        # polytope description as linear inequality
        self.A = None
        self.b = None
        self.bbox = None  # SVG standard uses same name
        self.vertices = None
        self.extreme = None  # extreme points of bounded polytope
        # internal details
        self._volume = None
        # Maximal inscribed ball
        self._radius = None
        self._center = None
        self._min_ball = None

    def __str__(self):
        """Return pretty-formatted H-representation."""

    def copy(self):
        pass

    def __hash__(self):
        # Necessary because `__eq__` is implemented.
        # Allows forming sets of polytope instances.
        return self(id)

    def __eq__(self):
        """Set equality (as set of N-dim points)."""

    def __ne__(self):
        """Set inequality."""

    def __le__(self):
        pass

    def __ge__(self):
        pass

    # TODO: should we use `min_ball` here?
    # This is how `fulldim` is defined.
    def __bool__(self):
        """Return `True` if polytope has nonzero volume."""

    __nonzero__ = __bool__

    # should implement `set` interface

    def __invert__(self):
        """Return a negated polytope.

        Set difference then becomes `a & ~ b`.
        """

    def __or__(self, other):
        """Set union."""

    def __and__(self, other):
        """Set intersection."""

    def __xor__(self, other):
        """Symmetric set difference."""

    def __sub__(self, other):
        """Set difference."""
        # same as `set` interface`
        #
        # for symmetry with Minkowski sum this would have
        # to be Minkowski difference, but how often do we
        # use Minkowski difference, and how often subtraction ?
        # Practicality beats purity [PEP 20].

    def __round__(self):
        """Convex hull.

        Copy of `self` for `Hull`, possibly a different set
        for a `Polytope`.
        """

    def __contains__(self, point):
        pass

    def contains(self, points):
        pass

    def union(self, others):
        """May return a `Polytope`. Support multiple arguments."""

    def intersection(self, others):
        pass

    def difference(self):
        """May return a `Polytope`."""

    def projection(self):
        pass

    def __truediv__(self, dim):
        # `/` used because it returns "how much of" the
        # numerator is made of the denominator.
        # So `poly / dim` makes some sense.
        return self.projection(dim)

    def translation(self):
        pass

    def __rshift__(self):
        # `@` is tempting, but `>>` evokes "move"
        # and a modification, whereas `@` evokes "place at",
        # which is an absolute operation.
        return self.translation()

    def rotation(self):
        pass

    def __matmul__(self):
        # we could allow homogeneous transformations too
        # `@` used because matrix multiplication with A
        return self.rotation()

    def scaling(self):
        pass

    def __mul__(self):
        return self.scaling()

    def reduction(self):
        """Remove redundant inequalities."""

    @property
    def dim(self):
        return np.shape(self.A)[1]

    @property
    def volume(self):
        pass

    def __abs__(self):
        """Return volume of polytope."""
        return self.volume

    @property
    def radius(self):
        """Maximal radius of any inscribed ball."""

    @property
    def center(self):
        """Center of a maximal inscribed ball."""

    @property
    def max_ball(self):
        """A maximal inscribed ball."""

    @property
    def bounding_box(self):
        pass  # PEP 8 naming: full words

    @property
    def min_ball(self):
        """A minimal circumscribed ball.

        There are no attributes `min_radius` and `min_center`,
        because common use cases involve points inside the
        polytope (e.g., annotating with some text).
        """

    @property
    def is_flat(self):
        # was `~ fulldim`, follows `str.isalpha`
        pass

    @property
    def is_convex(self):
        pass

    def plot(self):
        pass

    def text(self):
        pass

    @classmethod
    def from_box(self):
        pass


# TODO: maybe subclass from a common class and override ?
class Polytope(object):

    def __init__(self):
        self.hulls = None  # `list` of convex polytopes

    def __iter__(self):
        return iter(self.hulls)

    def __getitem__(self, key):
        return self.hulls[key]

    def __len__(self):
        return len(self.hulls)

    # Set-theoretic operators (union etc.) have similar
    # structure with logical operators (disjunction etc.),
    # and are actually defined in terms of them.
    # So it makes sense to use `__or__` for `union`.
    # It also matches `set`.

    # TODO: should we use `*` instead?
    def __add__(self, other):
        """Minkowski sum."""
