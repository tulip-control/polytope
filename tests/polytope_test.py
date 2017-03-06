#!/usr/bin/env python
"""Tests for the polytope subpackage."""
import logging

import numpy as np
from numpy.testing import assert_allclose
import polytope as pc
from polytope.polytope import _solve_rotation

log = logging.getLogger('polytope.polytope')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


# unit square in first quadrant
Ab = np.array([[0.0, 1.0, 1.0],
               [0.0, -1.0, 0.0],
               [1.0, 0.0, 1.0],
               [-1.0, 0.0, 0.0]])

# unit square in second quadrant
Ab2 = np.array([[-1.0, 0.0, 1.0],
               [1.0, 0.0, 0.0],
               [0.0, 1.0, 1.0],
               [0.0, -1.0, 0.0]])

# unit square in third quadrant
Ab3 = np.array([[0.0, 1.0, 0.0],
               [0.0, -1.0, 1.0],
               [1.0, 0.0, 0.0],
               [-1.0, 0.0, 1.0]])

# unit square in fourth quadrant
Ab4 = np.array([[0.0, 1.0, 0.0],
               [0.0, -1.0, 1.0],
               [1.0, 0.0, 1.0],
               [-1.0, 0.0, 0.0]])

A = Ab[:, 0:2]
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
    p4 = pc.qhull(np.array([[0, 0], [1, 0], [0, 1]]))
    assert(p4 == pc.Polytope(
        np.array([[1, 1], [0, -1], [0, -1]]),
        np.array([1, 0, 0])))


def region_rotation_test():
    p = pc.Region([pc.Polytope(A, b)])
    p1 = pc.Region([pc.Polytope(A, b)])
    p2 = pc.Region([pc.Polytope(Ab2[:, 0:2], Ab2[:, 2])])
    p3 = pc.Region([pc.Polytope(Ab3[:, 0:2], Ab3[:, 2])])
    p4 = pc.Region([pc.Polytope(Ab4[:, 0:2], Ab4[:, 2])])

    p = p.rotation(0, 1, np.pi/2)
    print(p.bounding_box)
    assert(p == p2)
    assert(not p == p3)
    assert(not p == p4)
    assert(not p == p1)
    assert_allclose(p.chebXc, [-0.5, 0.5])

    p = p.rotation(0, 1, np.pi/2)
    assert(p == p3)
    assert_allclose(p.chebXc, [-0.5, -0.5])

    p = p.rotation(0, 1, np.pi/2)
    assert(p == p4)
    assert_allclose(p.chebXc, [0.5, -0.5])

    p = p.rotation(0, 1, np.pi/2)
    assert(p == p1)
    assert_allclose(p.chebXc, [0.5, 0.5])


def polytope_rotation_test():
    p = pc.Polytope(A, b)
    p1 = pc.Polytope(A, b)
    p2 = pc.Polytope(Ab2[:, 0:2], Ab2[:, 2])
    p3 = pc.Polytope(Ab3[:, 0:2], Ab3[:, 2])
    p4 = pc.Polytope(Ab4[:, 0:2], Ab4[:, 2])

    p = p.rotation(0, 1, np.pi/2)
    print(p.bounding_box)
    assert(p == p2)
    assert(not p == p3)
    assert(not p == p4)
    assert(not p == p1)
    assert_allclose(p.chebXc, [-0.5, 0.5])

    p = p.rotation(0, 1, np.pi/2)
    assert(p == p3)
    assert_allclose(p.chebXc, [-0.5, -0.5])

    p = p.rotation(0, 1, np.pi/2)
    assert(p == p4)
    assert_allclose(p.chebXc, [0.5, -0.5])

    p = p.rotation(0, 1, np.pi/2)
    assert(p == p1)
    assert_allclose(p.chebXc, [0.5, 0.5])


def region_translation_test():
    p = pc.Region([pc.Polytope(A, b)])
    p1 = pc.Region([pc.Polytope(A, b)])
    p2 = pc.Region([pc.Polytope(Ab2[:, 0:2], Ab2[:, 2])])

    p = p.translation([-1, 0])
    assert(p == p2)
    assert(not p == p1)
    p = p.translation([1, 0])
    assert(p == p1)


def polytope_translation_test():
    p = pc.Polytope(A, b)
    p1 = pc.Polytope(A, b)
    p2 = pc.Polytope(Ab2[:, 0:2], Ab2[:, 2])

    p = p.translation([-1, 0])
    assert(p == p2)
    assert(not p == p1)
    p = p.translation([1, 0])
    assert(p == p1)


def solve_rotation_test_090(atol=1e-15):
    g1 = np.array([0, 1, 1, 0])
    g2 = np.array([0, 1, 0, 0])
    R = _solve_rotation(g1, g2)

    e0 = np.array([0, 1, 1, 1])
    e1 = np.array([0, 0, -1, 0])
    e2 = np.array([0, 0, 0, 0])

    t0 = np.array([0, 1, -1, 1])
    t1 = np.array([0, -1, 0, 0])
    t2 = np.array([0, 0, 0, 0])

    assert_allclose(R.dot(e0), t0, atol=atol)
    assert_allclose(R.dot(e1), t1, atol=atol)
    assert_allclose(R.dot(e2), t2, atol=atol)


def solve_rotation_test_180(atol=1e-15):
    g1 = np.array([0, 1, 0, 0])
    g2 = np.array([0, 0, 1, 0])
    R = _solve_rotation(g1, g2)

    e0 = np.array([0, 1, 1, 1])
    e1 = np.array([0, 0, -1, 0])
    e2 = np.array([0, 0, 0, 0])

    t0 = np.array([0, -1, -1, 1])
    t1 = np.array([0, 0, 1, 0])
    t2 = np.array([0, 0, 0, 0])

    assert_allclose(R.dot(e0), t0, atol=atol)
    assert_allclose(R.dot(e1), t1, atol=atol)
    assert_allclose(R.dot(e2), t2, atol=atol)


def solve_rotation_test_270R(atol=1e-15):
    g1 = np.array([0, -1, 0, 0])
    g2 = np.array([0, 1, 1, 0])
    R = _solve_rotation(g1, g2)

    e0 = np.array([0, 1, 1, 1])
    e1 = np.array([0, 0, -1, 0])
    e2 = np.array([0, 0, 0, 0])

    t0 = np.array([0, -1, 1, 1])
    t1 = np.array([0, 1, 0, 0])
    t2 = np.array([0, 0, 0, 0])

    assert_allclose(R.dot(e0), t0, atol=atol)
    assert_allclose(R.dot(e1), t1, atol=atol)
    assert_allclose(R.dot(e2), t2, atol=atol)


def solve_rotation_test_270L(atol=1e-15):
    g1 = np.array([0, -1, 0, 0])
    g2 = np.array([0, 1, -1, 0])
    R = _solve_rotation(g1, g2)

    e0 = np.array([0, 1, 1, 1])
    e1 = np.array([0, 0, -1, 0])
    e2 = np.array([0, 0, 0, 0])

    t0 = np.array([0, 1, -1, 1])
    t1 = np.array([0, -1, 0, 0])
    t2 = np.array([0, 0, 0, 0])

    assert_allclose(R.dot(e0), t0, atol=atol)
    assert_allclose(R.dot(e1), t1, atol=atol)
    assert_allclose(R.dot(e2), t2, atol=atol)


if __name__ == '__main__':
    comparison_test()
