#!/usr/bin/env python
"""Tests for projections of polytopes."""
import logging

import numpy as np

import polytope as pc


log = logging.getLogger('polytope.polytope')
log.setLevel(logging.INFO)


def test_fourier_motzkin_square():
    # Setup a square and project it on the x and y axis
    a = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ])
    b = np.array([
        -1.0,
        2.0,
        -1.0,
        2.0,
    ])
    poly = pc.Polytope(a, b)
    project_dim_0 = pc.polytope.projection_fm(poly, None, np.array([1]))
    project_dim_1 = pc.polytope.projection_fm(poly, None, np.array([0]))
    expected_a = np.array([[-1.0], [1.0]])
    expected_b = np.array([-1.0, 2.0])
    ind_0 = np.argsort(project_dim_0.A, axis=0).flatten()
    ind_1 = np.argsort(project_dim_1.A, axis=0).flatten()
    assert np.allclose(
        project_dim_0.A[ind_0],
        expected_a,
        pc.polytope.ABS_TOL),\
        (project_dim_0.A[ind_0], expected_a)
    assert np.allclose(
        project_dim_0.b[ind_0],
        expected_b,
        pc.polytope.ABS_TOL),\
        (project_dim_0.b[ind_0], expected_b)
    assert np.allclose(
        project_dim_1.A[ind_1],
        expected_a,
        pc.polytope.ABS_TOL),\
        (project_dim_1.A[ind_1], expected_a)
    assert np.allclose(
        project_dim_1.b[ind_1],
        expected_b,
        pc.polytope.ABS_TOL),\
        (project_dim_1.b[ind_1], expected_b)


def test_fourier_motzkin_triangle():
    # Setup a triangle and project it on the x and y axis.
    a = np.array([
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ])
    b = np.array([
        -1.0,
        4.0,
        0.0,
    ])
    poly = pc.Polytope(a, b)
    project_dim_0 = pc.polytope.projection_fm(poly, None, np.array([1]))
    project_dim_1 = pc.polytope.projection_fm(poly, None, np.array([0]))
    expected_a_0 = np.array([[-1.0], [1.0]])
    expected_b_0 = np.array([-1.0, 3.0])
    ind_0 = np.argsort(project_dim_0.A, axis=0).flatten()
    expected_a_1 = np.array([[-1.0], [1.0]])
    expected_b_1 = np.array([-1.0, 2.0])
    ind_1 = np.argsort(project_dim_1.A, axis=0).flatten()
    assert np.allclose(
        project_dim_0.A[ind_0],
        expected_a_0,
        pc.polytope.ABS_TOL), \
        (project_dim_0.A[ind_0], expected_a_0)
    assert np.allclose(
        project_dim_0.b[ind_0],
        expected_b_0,
        pc.polytope.ABS_TOL), \
        (project_dim_0.b[ind_0], expected_b_0)
    assert np.allclose(
        project_dim_1.A[ind_1],
        expected_a_1,
        pc.polytope.ABS_TOL), \
        (project_dim_1.A[ind_1], expected_a_1)
    assert np.allclose(
        project_dim_1.b[ind_1],
        expected_b_1,
        pc.polytope.ABS_TOL), \
        (project_dim_1.b[ind_1], expected_b_1)


def test_projection_iterhull():
    # This is a unit cube with vertices at (0,0,0), (1,0,0), (0,1,0), ...
    p = pc.Polytope(
        A=np.array(
            [
                [1.0, -0.0, 0.0],
                [-0.0, -0.0, -1.0],
                [-0.0, 1.0, 0.0],
                [1.0, 0.0, -0.0],
                [-0.0, -1.0, -0.0],
                [-0.0, -0.0, 1.0],
                [-0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [-0.0, -1.0, 0.0],
                [-0.0, 1.0, -0.0],
                [-0.0, -0.0, 1.0],
                [-1.0, -0.0, -0.0],
            ]
        ),
        b=np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
    )
    q = p.project([1, 2], solver="iterhull")

    expected_q_A = np.array([[1.0, -0.0], [-0.0, -1.0], [0.0, 1.0], [-1.0, -0.0]])
    expected_q_b = np.array([1.0, 0.0, 1.0, 0.0])

    assert np.all(q.A.shape == expected_q_A.shape)
    assert np.all(q.b.shape == expected_q_b.shape)
    if not np.allclose(q.A, expected_q_A):
        # Due to randomization, the A matrix found in a given test
        # execution may have rows permuted from the expected A matrix.
        # Therefore, search for a permutation before failing the test.
        permutation = [-1, -1, -1, -1]
        for expected_i, expected_row in enumerate(expected_q_A):
            actual_i = 0
            for actual_i, actual_row in enumerate(q.A):
                if np.allclose(expected_row, actual_row):
                    assert actual_i not in permutation
                    permutation[expected_i] = actual_i
                    break
            assert permutation[expected_i] >= 0, "projection is missing expected row"
        assert np.allclose(expected_q_b, q.b[permutation])
    else:
        assert np.allclose(expected_q_b, q.b)
