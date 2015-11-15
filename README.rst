Polytope
========

This is the source repository for ``polytope``, a toolbox for geometric
operations on polytopes in any dimension.  Documentation and examples are
available in the directories ``doc/`` and ``examples/``.


Installation
------------

From `PyPI <https://pypi.python.org/pypi/polytope>`_::

  pip install polytope

From source::

  python setup.py install

If this fails, because ``setuptools`` attempts to install
``scipy`` before ``numpy``, then::

  pip install .


Dependencies
------------
Required: ``numpy``, ``scipy``, ``networkx``.
Optionally, if ``cvxopt`` is installed and
linked to `GLPK <https://en.wikipedia.org/wiki/GNU_Linear_Programming_Kit>`_,
then ``polytope`` will prefer GLPK,
because it is faster than ``scipy``.
For more details, see ``requirements.txt``.


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
