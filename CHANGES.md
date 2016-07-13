# Polytope package update history


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
