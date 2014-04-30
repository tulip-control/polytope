#!/usr/bin/env python
"""
Tests for the polytope subpackage.
"""
import numpy as np
import polytope as pc

# unit square
Ab = np.array([[0.0, 1.0, 1.0],
               [0.0, -1.0, 0.0],
               [1.0, 0.0, 1.0],
               [-1.0, 0.0, 0.0]])

A = Ab[:,0:2]
b = Ab[:, 2]

def comparison_test():
    p = pc.Polytope(A=A, b=b)
    p2 = pc.Polytope(A=A, b=2*b)
    
    assert(p <= p2)
    assert(not p2 <= p)
    assert(not p2 == p)
