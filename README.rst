Polytope
========
This is the source repository for ``polytope``, a toolbox for geometric
operations on polytopes in any dimension.

Installation
------------

It suffices to::

  python setup.py install

To avoid checking for optional dependencies, add the option "nocheck"::

  python setup.py install nocheck

Dependencies
------------
Required: ``numpy``, ``cvxopt``, ``networkx``.

Optional: ``matplotlib``, ``scipy``.

License
-------
Polytope is licensed under the 3-clause BSD license.  The full statement is
provided in the file named `LICENSE`.

Acknowledgment
--------------
Polytope was part of the `Temporal Logic Planning Toolbox (TuLiP)
<http://tulip-control.org>`_ before growing to become an independent package.
It originates from changesets 7bb73a9f725572db454a0a5e4957da84bc778f65 and
3178c570ee1ef06eb8ace033f205f51743ac54c6 of `TuLiP
<https://github.com/tulip-control/tulip-control>`_.
