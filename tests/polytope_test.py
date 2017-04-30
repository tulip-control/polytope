#!/usr/bin/env python
"""Tests for the polytope subpackage."""
import logging

import numpy as np
from numpy.testing import assert_allclose
import polytope as pc
from polytope.polytope import solve_rotation_ap, givens_rotation_matrix

log = logging.getLogger('polytope.polytope')
log.setLevel(logging.INFO)


class operations_test(object):
    def setUp(self):
        # unit square in first quadrant
        self.Ab = np.array([[0.0, 1.0, 1.0],
                            [0.0, -1.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [-1.0, 0.0, 0.0]])

        # unit square in second quadrant
        self.Ab2 = np.array([[-1.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0],
                             [0.0, -1.0, 0.0]])

        # unit square in third quadrant
        self.Ab3 = np.array([[0.0, 1.0, 0.0],
                             [0.0, -1.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [-1.0, 0.0, 1.0]])

        # unit square in fourth quadrant
        self.Ab4 = np.array([[0.0, 1.0, 0.0],
                             [0.0, -1.0, 1.0],
                             [1.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0]])

        self.A = self.Ab[:, 0:2]
        self.b = self.Ab[:, 2]

    def tearDown(self):
        pass


    def comparison_test(self):
        p = pc.Polytope(self.A, self.b)
        p2 = pc.Polytope(self.A, 2*self.b)

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


    def region_rotation_test(self):
        p = pc.Region([pc.Polytope(self.A, self.b)])
        p1 = pc.Region([pc.Polytope(self.A, self.b)])
        p2 = pc.Region([pc.Polytope(self.Ab2[:, 0:2], self.Ab2[:, 2])])
        p3 = pc.Region([pc.Polytope(self.Ab3[:, 0:2], self.Ab3[:, 2])])
        p4 = pc.Region([pc.Polytope(self.Ab4[:, 0:2], self.Ab4[:, 2])])

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


    def polytope_rotation_test(self):
        p = pc.Polytope(self.A, self.b)
        p1 = pc.Polytope(self.A, self.b)
        p2 = pc.Polytope(self.Ab2[:, 0:2], self.Ab2[:, 2])
        p3 = pc.Polytope(self.Ab3[:, 0:2], self.Ab3[:, 2])
        p4 = pc.Polytope(self.Ab4[:, 0:2], self.Ab4[:, 2])

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


    def region_translation_test(self):
        p = pc.Region([pc.Polytope(self.A, self.b)])
        p1 = pc.Region([pc.Polytope(self.A, self.b)])
        p2 = pc.Region([pc.Polytope(self.Ab2[:, 0:2], self.Ab2[:, 2])])

        p = p.translation([-1, 0])
        assert(p == p2)
        assert(not p == p1)
        p = p.translation([1, 0])
        assert(p == p1)


    def polytope_translation_test(self):
        p = pc.Polytope(self.A, self.b)
        p1 = pc.Polytope(self.A, self.b)
        p2 = pc.Polytope(self.Ab2[:, 0:2], self.Ab2[:, 2])

        p = p.translation([-1, 0])
        assert(p == p2)
        assert(not p == p1)
        p = p.translation([1, 0])
        assert(p == p1)

    def region_empty_test(self):
        # Note that as of commit a037b555758ed9ee736fa7cb324d300b8d622fb4
        # Region.__init__ deletes empty polytopes from
        # the given list of polytopes at instantiation.
        reg = pc.Region()
        reg.list_poly = [pc.Polytope(), pc.Polytope()]
        assert len(reg) > 0
        assert pc.is_empty(reg)

    def polytope_full_dim_test(self):
        assert pc.is_fulldim(pc.Polytope(self.A, self.b))
        assert pc.is_fulldim(pc.Polytope(self.Ab2[:, 0:2], self.Ab2[:, 2]))
        assert not pc.is_fulldim(pc.Polytope())
        assert not pc.is_fulldim(pc.Polytope(self.A, self.b - 1e3))

    def region_full_dim_test(self):
        assert not pc.is_fulldim(pc.Region())

        p1 = pc.Polytope(self.A, self.b)
        p2 = pc.Polytope(self.Ab2[:, 0:2], self.Ab2[:, 2])
        reg = pc.Region([p1, p2])
        assert pc.is_fulldim(reg)

        # Adding empty polytopes should not affect the
        # full-dimensional status of this region.
        reg.list_poly.append(pc.Polytope())
        assert pc.is_fulldim(reg)
        reg.list_poly.append(pc.Polytope(self.A, self.b - 1e3))
        assert pc.is_fulldim(reg)

    def polytope_intersect_test(self):
        p1 = pc.Polytope(self.A, self.b)
        p2 = pc.Polytope(self.Ab2[:, 0:2], self.Ab2[:, 2])
        p3 = p1.intersect(p2)
        assert pc.is_fulldim(p1)
        assert pc.is_fulldim(p2)
        assert not pc.is_fulldim(p3)

        # p4 is the unit square with center at the origin.
        p4 = pc.Polytope(np.array([[ 1.,  0.],
                                   [ 0.,  1.],
                                   [-1.,  0.],
                                   [ 0., -1.]]),
                         np.array([0.5, 0.5, 0.5, 0.5]))
        p5 = p2.intersect(p4)
        assert pc.is_fulldim(p4)
        assert pc.is_fulldim(p5)



def solve_rotation_test_090(atol=1e-15):
    g1 = np.array([0, 1, 1, 0])
    g2 = np.array([0, 1, 0, 0])
    R = solve_rotation_ap(g1, g2)

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
    R = solve_rotation_ap(g1, g2)

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
    R = solve_rotation_ap(g1, g2)

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
    R = solve_rotation_ap(g1, g2)

    e0 = np.array([0, 1, 1, 1])
    e1 = np.array([0, 0, -1, 0])
    e2 = np.array([0, 0, 0, 0])

    t0 = np.array([0, 1, -1, 1])
    t1 = np.array([0, -1, 0, 0])
    t2 = np.array([0, 0, 0, 0])

    assert_allclose(R.dot(e0), t0, atol=atol)
    assert_allclose(R.dot(e1), t1, atol=atol)
    assert_allclose(R.dot(e2), t2, atol=atol)


def givens_rotation_test_180(atol=1e-15):
    R = givens_rotation_matrix(1, 2, np.pi, 4)

    e0 = np.array([0, 1, 1, 1])
    e1 = np.array([0, 0, -1, 0])
    e2 = np.array([0, 0, 0, 0])

    t0 = np.array([0, -1, -1, 1])
    t1 = np.array([0, 0, 1, 0])
    t2 = np.array([0, 0, 0, 0])

    assert_allclose(R.dot(e0), t0, atol=atol)
    assert_allclose(R.dot(e1), t1, atol=atol)
    assert_allclose(R.dot(e2), t2, atol=atol)


def givens_rotation_test_270L(atol=1e-15):
    g1 = np.array([0, -1, 0, 0])
    g2 = np.array([0, 1, -1, 0])
    R = givens_rotation_matrix(1, 2, 3*np.pi/2, 4)

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
    pass
