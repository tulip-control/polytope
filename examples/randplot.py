#!/usr/bin/env python
"""
Sample N points in the unit square, compute hull and plot.

  Usage: randplot.py [N]

The default value of N is 3.  Note that plotting requires matplotlib
(http://matplotlib.org), which is an optional dependency.
"""

import polytope

import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        N = 3
    else:
        N = int(sys.argv[1])

    V = np.random.rand(N, 2)
    
    print("Sampled "+str(N)+" points:")
    print(V)

    P = polytope.qhull(V)
    print("Computed the convex hull:")
    print(P)
    
    V_min = polytope.extreme(P)
    print("which has extreme points:")
    print(V_min)

    P.plot()
    plt.show()
