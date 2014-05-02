#!/usr/bin/env python
"""
Tests for the polytope module
"""
import numpy as np
import polytope as pc

# [0, 1] x [0, 1]
A = np.array([
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0]
])

b = np.array([[1.0, 0.0, 1.0, 0.0]])

# [0, 0.5] x [0, 0.5]
A1 = np.array([
    [0.0, 2.0],
    [0.0, -1.0],
    [2.0, 0.0],
    [-1.0, 0.0]
])

b1 = np.array([1.0, 0.0, 1.0, 0.0])

# [1, 2] x [0, 1]
A2 = np.array([
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0]
])

b2 = np.array([1.0, 0.0, 2.0, -1.0])

# [2, 3] x [0, 1]
A3 = np.array([
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0]
])

b3 = np.array([1.0, 0.0, 3.0, -2.0])

def plot_them_to_be_sure():
    p = pc.Polytope(A=A, b=b)
    ax = p.plot()
    ax.axis('tight')
    
    p = pc.Polytope(A=A1, b=b1)
    ax = p.plot()
    ax.axis('tight')
    
    p = pc.Polytope(A=A2, b=b2)
    ax = p.plot()
    ax.axis('tight')

def polytope_init_test():
    p = pc.Polytope(A=A, b=b)
    
    assert(len(p) == 1)
    assert(np.allclose(p[0].A, A) )
    assert(np.allclose(p[0].b, b) )
    
    assert(p._bbox is None)
    assert(p._x is None)
    assert(p._r is None)
    assert(p._bbox is None)
    assert(p._fulldim is None)
    assert(p._volume is None)
    assert(p._abs_tol == pc.polytope.ABS_TOL)
    
    p = pc.Polytope([(A, b), (A2, b2)])
    
    assert(len(p) == 2)
    
    assert(np.allclose(p[0].A, A) )
    assert(np.allclose(p[0].b, b) )
    
    assert(np.allclose(p[1].A, A2) )
    assert(np.allclose(p[1].b, b2) )
    
    p = pc.Polytope([(A, b), (A2, b2)], A=A3, b=b3)
    
    assert(len(p) == 3)
    
    assert(np.allclose(p[0].A, A) )
    assert(np.allclose(p[0].b, b) )
    
    assert(np.allclose(p[1].A, A2) )
    assert(np.allclose(p[1].b, b2) )
    
    assert(np.allclose(p[2].A, A3) )
    assert(np.allclose(p[2].b, b3) )

def contains_test():
    p = pc.Polytope(A=A, b=b)
    
    assert([0, 0] in p)
    assert([0, 1] in p)
    assert([1, 0] in p)
    assert([1, 1] in p)

def diff_test():
    big = pc.Polytope(A=A, b=b)
    small = pc.Polytope(A=A1, b=b1)
    
    diff_0 = big - small
    diff_1 = big.diff(small)
    
    assert(np.allclose(diff_0[0].A, diff_1[0].A) )
    assert(np.allclose(diff_0[0].b, diff_1[0].b) )
    
    assert(diff_0 == diff_1)
    
    assert([0, 0] not in diff_0)
    assert([0, 0.5] in diff_0)
    assert([0.5, 0] in diff_0)
    assert([0, 1] in diff_0)
    assert([1, 0] in diff_0)
    assert([1, 1] in diff_0)
    
    assert(diff_0 <= big)
    assert(not diff_0 <= small)

def union_test():
    p0 = pc.Polytope(A=A, b=b)
    p1 = pc.Polytope(A=A2, b=b2)
    
    union = p0.union(p1, check_convex=True)
    
    assert(len(union) == 1)
    
    assert([0, 0] in union)
    assert([2, 0] in union)
    assert([2, 1] in union)
    assert([0, 1] in union)
    
    assert(p0 <= union)
    assert(not p0 >= union)
    
    assert(p1 <= union)
    assert(not p1 >= union)
    
    union_1 = p0 + p1
    
    assert(len(union) == 1)
    
    # caution: errors here might be caused only by
    # a permutation of the rows
    Au = np.array([
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0]
    ])
    
    bu = np.array([0.0, 1.0, 0.0, 2.0])
    
    assert(np.allclose(union[0].A, np.array(Au) ) )
    assert(np.allclose(union[0].b, np.array(bu) ) )
    
    assert(union_1 == union)

def intersection_test():
    p0 = pc.Polytope(A=A, b=b)
    p1 = pc.Polytope(A=A1, b=b1)
    
    isect = p0.intersection(p1)
    
    print(len(isect))
    print(isect)
    assert(len(isect) == 1)
    
    assert(isect == p1)
    assert(isect <= p1)
    assert(isect <= p0)
    
    isect_1 = p0 & p1
    
    assert(isect_1 == isect)

def comparison_test():
    p = pc.Polytope(A=A, b=b)
    p2 = pc.Polytope(A=A, b=2*b)
    
    assert(p <= p2)
    assert(not p2 <= p)
    assert(not p2 == p)

if __name__ == '__main__':
    plot_them_to_be_sure()
