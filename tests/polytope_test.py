#!/usr/bin/env python
"""
Tests for the polytope subpackage.
"""
import logging

import numpy as np

log = logging.getLogger('polytope.polytope')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

import polytope as pc

# unit square
Ab = np.array([[0.0, 1.0, 1.0],
               [0.0, -1.0, 0.0],
               [1.0, 0.0, 1.0],
               [-1.0, 0.0, 0.0]])

A = Ab[:,0:2]
b = Ab[:, 2]

def comparison_test():
    p = pc.Polytope(A, b)
    p2 = pc.Polytope(A, 2*b)

    assert(p <= p2)
    assert(not p2 <= p)
    assert(not p2 == p)

    r = pc.Region([p])
    r2 = pc.Region([p2])

    assert(r <= r2)
    assert(not r2 <= r)
    assert(not r2 == r)

    # test H-rep -> V-rep -> H-rep
    v = pc.extreme(p)
    p3 = pc.qhull(v)
    assert(p3 == p)

    # test V-rep -> H-rep with d+1 points
    p4 = pc.qhull(np.array([[0,0], [1,0], [0,1]]))
    assert(p4 == pc.Polytope(np.array([[1,1], [0,-1], [0,-1]]), np.array([1,0,0])))


if __name__ == '__main__':
    comparison_test()
