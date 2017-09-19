"""Interface to linear programming solvers.

By default, for linear programming the `polytope` package selects
the fastest solver that it finds installed. You can change this
default by setting the variable `lp_solver` in the module
`_solvers`. For example:

```python
from polytope import _solvers

_solvers.lp_solver = 'scipy'
```

Choose an installed solver to avoid errors.
"""
# Copyright (c) 2011-2017 by California Institute of Technology
# All rights reserved. Licensed under BSD-3.
#
from __future__ import absolute_import
import logging

import numpy as np
from scipy import optimize


logger = logging.getLogger(__name__)
try:
    from cvxopt import matrix, solvers
    import cvxopt.glpk
    lp_solver = 'glpk'
    # Hide optimizer output
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
    logger.info('will use `cvxopt.glpk` solver')
except ImportError:
    lp_solver = 'scipy'
    logger.warn(
        '`polytope` failed to import `cvxopt.glpk`.\n'
        'Will use `scipy.optimize.linprog`.')



def lpsolve(c, G, h, solver=None):
    """Try to solve linear program with `cvxopt.glpk`, else `scipy`.

    Solvers:
        - `cvxopt.glpk`: identified by `'glpk'`
        - `scipy.optimize.linprog`: identified by `'scipy'`

    @param solver:
        - `in {'glpk', 'scipy'}`
        - `None`: use the fastest installed solver,
          as follows:

            1. use GLPK if installed
            2. otherwise use SciPy

        You can change the default choice of solver by setting
        the module variable `lp_solver`. See the module's
        docstring for an example.

    @return: solution with status as in `scipy.optimize.linprog`
    @rtype: `dict(status=int, x=argmin, fun=min_value)`
    """
    if solver is None:
        solver = lp_solver  # choose fastest installed solver
    if solver == 'glpk' and lp_solver != 'glpk':
        raise ImportError('GLPK requested but failed to import.')
    if solver == 'glpk':
        result = _solve_lp_using_glpk(c, G, h)
    elif solver == 'scipy':
        result = _solve_lp_using_scipy(c, G, h)
    else:
        raise Exception(
            'unknown LP solver "{s}".'.format(s=lp_solver))
    return result


def _solve_lp_using_glpk(c, G, h, A=None, b=None):
    """Attempt linear optimization using `cvxopt.glpk`."""
    assert lp_solver == 'glpk', 'GLPK failed to import'
    if A is not None:
        A = matrix(A)
    if b is not None:
        b = matrix(b)
    sol = solvers.lp(
        c=matrix(c), G=matrix(G), h=matrix(h),
        A=A, b=b, solver='glpk')
    result = dict()
    if sol['status'] == 'optimal':
        result['status'] = 0
    elif sol['status'] == 'primal infeasible':
        result['status'] = 2
    elif sol['status'] == 'dual infeasible':
        result['status'] = 3
    elif sol['status'] == 'unknown':
        result['status'] = 4
    else:
        raise ValueError((
            '`cvxopt.solvers.lp` returned unexpected '
            'status value: {v}').format(v=sol['status']))
    # `cvxopt.solvers.lp` returns an array of shape `(2, 1)`
    # squeeze only the second dimension, to obtain a 1-D array
    # thus match what `scipy.optimize.linprog` returns.
    x = sol['x']
    if x is not None:
        assert x.typecode == 'd', x.typecode
        result['x'] = np.fromiter(x, dtype=np.double)
    else:
        result['x'] = None
    result['fun'] = sol['primal objective']
    return result


def _solve_lp_using_scipy(c, G, h):
    """Attempt linear optimization using `scipy.optimize.linprog`."""
    sol = optimize.linprog(
        c, G, np.transpose(h),
        None, None, bounds=(None, None))
    return dict(
        status=sol.status,
        x=sol.x,
        fun=sol.fun)
