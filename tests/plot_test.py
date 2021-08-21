#!/usr/bin/env python
"""Tests for plotting."""
import unittest

try:
    import matplotlib as mpl
    import matplotlib.patches
except ImportError:
    mpl = None

import polytope as pc
from polytope import plot


class Axes(object):
    """Mock class."""

    def add_patch(self, x):
        pass


@unittest.skipIf(
    mpl is None,
    '`matplotlib` is not installed')
def test_plot_transition_arrow():
    p0 = pc.box2poly([[0.0, 1.0], [0.0, 2.0]])
    p1 = pc.box2poly([[0.1, 2.0], [0.0, 2.0]])
    # matplotlib.patches is loaded also by matplotlib.pyplot
    # and .figures, so instantiating real Axes w/o
    # loading patches is impossible
    ax = Axes()
    arrow = plot.plot_transition_arrow(p0, p1, ax=ax)
    assert(isinstance(arrow, matplotlib.patches.Arrow))
