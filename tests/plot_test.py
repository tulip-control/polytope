#!/usr/bin/env python
"""
Tests for plotting.
"""
import polytope as pc
from polytope import plot

# fake
class Axes(object):
    def add_patch(self, x):
        pass

def test_plot_transition_arrow():
    p0 = pc.box2poly([[0.0, 1.0], [0.0, 2.0]])
    p1 = pc.box2poly([[0.1, 2.0], [0.0, 2.0]])

    # matplotlib.patches is loaded also by matplotlib.pyplot
    # and .figures, so instantiating real Axes w/o
    # loading patches is impossible
    ax = Axes()
    arrow = plot.plot_transition_arrow(p0, p1, ax=ax)

    import matplotlib.patches # also loaded by plt
    assert(isinstance(arrow, matplotlib.patches.Arrow))
