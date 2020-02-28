# Copyright (c) 2011-2017 by California Institute of Technology
# All rights reserved. Licensed under 3-clause BSD.
"""Interface to linear programming solvers.

The `polytope` package selects the default solver as follows:

1. use GLPK if installed
2. otherwise use SciPy

You can change this default at runtime by setting the variable
`default_solver` in the module `solvers`.

For example:

```python
from polytope import solvers

solvers.default_solver = 'scipy'

# to inspect which solvers were successfully imported:
print(solvers.installed_solvers)
```

Choose an installed solver to avoid errors.
"""
from __future__ import absolute_import
import logging

import numpy as np
from scipy import optimize


logger = logging.getLogger(__name__)
installed_solvers = {'scipy'}
try:
    import cvxopt as cvx
    import cvxopt.glpk
    from cvxopt import matrix

    installed_solvers.add('glpk')
    # Hide optimizer output
    cvx.solvers.options['show_progress'] = False
    cvx.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
except ImportError:
    logger.warning(
        '`polytope` failed to import `cvxopt.glpk`.')
try:
    import mosek
    installed_solvers.add('mosek')
except ImportError:
    logger.info('MOSEK solver not found.')


# choose default from installed choices
if 'glpk' in installed_solvers:
    default_solver = 'glpk'
elif 'scipy' in installed_solvers:
    default_solver = 'scipy'
    logger.warning('will use `scipy.optimize.linprog`')
else:
    raise ValueError(
        "`installed_solvers` wasn't empty above?")



def lpsolve(c, G, h, solver=None):
    """Try to solve linear program with given or default solver.

    Solvers:
        - `cvxopt.glpk`: identified by `'glpk'`
        - `scipy.optimize.linprog`: identified by `'scipy'`
        - MOSEK: identified by `'mosek'`

    @param solver:
        - `in {'glpk', 'mosek', 'scipy'}`
        - `None`: use the module's `default_solver`

        You can change the default choice of solver by setting
        the module variable `default_solver`. See the module's
        docstring for an example.

    @return: solution with status as in `scipy.optimize.linprog`
    @rtype: `dict(status=int, x=argmin, fun=min_value)`
    """
    if solver is None:
        solver = default_solver
    if solver == 'glpk' or solver == 'mosek':
        result = _solve_lp_using_cvxopt(c, G, h, solver=solver)
    elif solver == 'scipy':
        result = _solve_lp_using_scipy(c, G, h)
    else:
        raise Exception(
            'unknown LP solver "{s}".'.format(s=solver))
    return result


def _solve_lp_using_cvxopt(c, G, h, A=None, b=None, solver='glpk'):
    """Attempt linear optimization using `cvxopt.glpk` or MOSEK.

    @param solver: `in {'glpk', 'mosek'}`
    """
    _assert_have_solver(solver)
    if A is not None:
        A = matrix(A)
    if b is not None:
        b = matrix(b)
    sol = cvx.solvers.lp(
        c=matrix(c), G=matrix(G), h=matrix(h),
        A=A, b=b, solver=solver)
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
    _assert_have_solver('scipy')
    sol = optimize.linprog(
        c, G, np.transpose(h),
        None, None, bounds=(None, None))
    return dict(
        status=sol.status,
        x=sol.x,
        fun=sol.fun)


def _assert_have_solver(solver):
    """Raise `RuntimeError` if `solver` is absent."""
    if solver in installed_solvers:
        return
    raise RuntimeError((
        'solver {solver} not in '
        'installed solvers: {have}').format(
            solver=solver, have=installed_solvers))
