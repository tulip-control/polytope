#!/usr/bin/env python
"""Tests for the polytope subpackage."""
import logging

from nose import tools as nt
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import scipy.optimize

import polytope as pc
from polytope.polytope import solve_rotation_ap, givens_rotation_matrix
from polytope import solvers

log = logging.getLogger('polytope.polytope')
log.setLevel(logging.INFO)


def test_polytope_str():
    # 1 constaint (so uniline)
    A = np.array([[1]])
    b = np.array([1])
    p = pc.Polytope(A, b)
    s = str(p)
    s_ = 'Single polytope \n  [[1.]] x <= [[1.]]\n'
    assert s == s_, (s, s_)
    # > 1 constraints (so multiline)
    polys = dict(
        p1d=[[0, 1]],
        p2d=[[0, 1], [0, 2]],
        p3d=[[0, 1], [0, 2], [0, 3]])
    strings = dict(
        p1d='Single polytope \n  [[ 1.] x <= [[1.]\n   [-1.]]|     [0.]]\n',
        p2d=(
            'Single polytope \n  [[ 1.  0.] |    [[1.]\n   [ 0.  1.] '
            'x <=  [2.]\n   [-1. -0.] |     [0.]\n   [-0. -1.]]|'
            '     [0.]]\n'),
        p3d=(
            'Single polytope \n  [[ 1.  0.  0.] |    [[1.]\n   '
            '[ 0.  1.  0.] |     [2.]\n   [ 0.  0.  1.] x <=  [3.]\n'
            '   [-1. -0. -0.] |     [0.]\n   [-0. -1. -0.] |'
            '     [0.]\n   [-0. -0. -1.]]|     [0.]]\n'))
    for name, poly in polys.items():
        p = pc.Polytope.from_box(poly)
        s = str(p)
        s_ = strings[name]
        assert s == s_, (s, s_)


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

    def polytope_contains_test(self):
        p = pc.Polytope(self.A, self.b)
        # single point
        point_i = [0.1, 0.3]
        point_o = [2, 0]
        assert point_i in p
        assert point_o not in p
        # multiple points
        many_points_i = np.random.random((2, 8))
        many_points_0 = np.random.random((2, 8)) - np.array([[0], [1]])
        many_points = np.concatenate([many_points_0, many_points_i], axis=1)
        truth = np.array([False] * 8 + [True] * 8, dtype=bool)
        assert_array_equal(p.contains(many_points), truth)

    def region_contains_test(self):
        A = np.array([[1.0],
                      [-1.0]])
        b = np.array([1.0, 0.0])
        poly = pc.Polytope(A, b)
        polys = [poly]
        reg = pc.Region(polys)
        assert 0.5 in reg
        # small positive tolerance (includes boundary)
        points = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]])
        c = reg.contains(points)
        c_ = np.array(
            [[False, True, True, True, False]], dtype=bool)
        # zero tolerance (excludes boundary)
        points = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]])
        c = reg.contains(points, abs_tol=0)
        c_ = np.array(
            [[False, False, True, False, False]], dtype=bool)
        assert np.all(c == c_), c

    def is_inside_test(self):
        box = [[0.0, 1.0], [0.0, 2.0]]
        p = pc.Polytope.from_box(box)
        point = np.array([0.0, 1.0])
        abs_tol = 0.01
        assert pc.is_inside(p, point)
        assert pc.is_inside(p, point, abs_tol)
        region = pc.Region([p])
        assert pc.is_inside(region, point)
        assert pc.is_inside(region, point, abs_tol)
        point = np.array([2.0, 0.0])
        assert not pc.is_inside(p, point)
        assert not pc.is_inside(p, point, abs_tol)
        region = pc.Region([p])
        assert not pc.is_inside(region, point)
        assert not pc.is_inside(region, point, abs_tol)
        abs_tol = 1.2
        assert pc.is_inside(p, point, abs_tol)
        assert pc.is_inside(region, point, abs_tol)


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


def test_lpsolve():
    # Ensure same API for both `scipy` and `cvxopt`.
    # Ensured by the different testing configurations.
    # Could change `polytope.polytope.default_solver` to
    # achieve the same result, when `cvxopt.glpk` is present.
    #
    # 2-D example
    c = np.array([1, 1], dtype=float)
    A = np.array([[-1, 0], [0, -1]], dtype=float)
    b = np.array([1, 1], dtype=float)
    res = solvers.lpsolve(c, A, b)
    x = res['x']
    assert x.ndim == 1, x.ndim
    assert x.shape == (2,), x.shape
    #
    # 1-D example
    c, A, b = example_1d()
    res = solvers.lpsolve(c, A, b)
    x = res['x']
    assert x.ndim == 1, x.ndim
    assert x.shape == (1,), x.shape


def example_1d():
    c = np.array([1], dtype=float)
    A = np.array([[-1]], dtype=float)
    b = np.array([1], dtype=float)
    return c, A, b


def test_lpsolve_solver_selection_scipy():
    # should always work, because `polytope` requires `scipy`
    c, A, b = example_1d()
    r_ = np.array([-1.0])
    # call directly to isolate from selection within `lpsolve`
    r = solvers._solve_lp_using_scipy(c, A, b)
    assert r['x'] == r_, r['x']
    r = solvers.lpsolve(c, A, b, solver='scipy')
    assert r['x'] == r_, r['x']


def test_lpsolve_solver_selection_glpk_present():
    c, A, b = example_1d()
    have_glpk = is_glpk_present()
    # skip if GLPK fails to import
    if not have_glpk:
        log.info(
            'Skipping GLPK test of `lpsolve` '
            'because GLPK failed to import, '
            'so assume not installed.')
        return
    r = solvers.lpsolve(c, A, b, solver='glpk')
    assert r['x'] == np.array([-1.0]), r['x']


def test_lpsolve_solver_selection_glpk_absent():
    c, A, b = example_1d()
    have_glpk = is_glpk_present()
    # skip if GLPK imports
    if have_glpk:
        log.info(
            'Skipping GLPK failure test, '
            'because GLPK is present.')
        return
    with nt.assert_raises(RuntimeError):
        solvers.lpsolve(c, A, b, solver='glpk')


def test_request_glpk_after_changing_default_to_scipy():
    c, A, b = example_1d()
    have_glpk = is_glpk_present()
    if not have_glpk:
        return
    assert solvers.default_solver != 'scipy'
    solvers.default_solver = 'scipy'
    solvers.lpsolve(c, A, b, solver='glpk')


def is_glpk_present():
    """Return `True` if `cvxopt.glpk` imports."""
    try:
        import cvxopt.glpk
        assert 'glpk' in solvers.installed_solvers, (
            solvers.installed_solvers)
        return True
    except ImportError:
        assert 'glpk' not in solvers.installed_solvers, (
            solvers.installed_solvers)
        return False


if __name__ == '__main__':
    pass
