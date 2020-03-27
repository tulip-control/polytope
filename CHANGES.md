# Polytope package update history


## version 0.2.2 - 27 March 2020

- customizable plotting in methods `Polytope.plot` and `Region.plot`


## version 0.2.1 - 24 November 2017

- rename method to `Polytope.contains`, was `are_inside`
- add method `contains` to `Region`
- deprecate function `polytope.is_inside`,
  use `in` and `contains` instead
- add arg `solver` to function `lpsolve`
- refactor by introducing new module `solvers`
- support MOSEK as solver, via function `cvxopt.solvers.lp`
- require `numpy >= 1.10.0`
- require `matplotlib >= 2.0.0` for tests


## version 0.2.0 - 7 July 2017

- negate `numpy.array` with operator `~`


## version 0.1.4 - 6 May 2017

- classes `polytope.Polytope`, `polytope.Region`:
  - add methods `translation`, `rotation`
- require `setuptools >= 23.0.0`
- require `numpy >= 1.7.1`
- require `scipy >= 0.18.0`


## version 0.1.3 - 31 August 2016

- support Python 3
- silence `cvxopt >= 1.1.8`


## version 0.1.2 - 12 July 2016

- require `scipy >= 0.16`
- use `scipy.optimize.linprog` if `cvxopt.glpk` fails to import
- PEP440-compliant version identifier `vX.Y.Z.dev0+commithash`
- test on Travis CI


## version 0.1.1 - 25 October 2015

- silence GLPK solver's output in `cvxopt`
- version that includes commit hash, when available
- define version in `polytope/version.py`


## version 0.1.0 - 27 April 2014

This initial release has very few changes since its break from TuLiP,
and is primarily intended to provide a reference version that easily integrates
with legacy code developed assuming polytope is a part of TuLiP.
