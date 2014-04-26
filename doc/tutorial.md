# Polytope package (brief) tutorial

First import the package:

```python
import polytope as pc
```

The `polytope` package is structured around two classes:

  - `Polytope`: a single convex polytope internally stored in
    [H-representation](https://en.wikipedia.org/wiki/Convex_polytope#Intersection_of_half-spaces)
    `A x <= b` where `A` is an `m x n` matrix and `b` and `m x 1` column matrix.
    `x` is a vector of `n` coordinates in `E^n` (Euclidean n-space).
  - `Region`: a container of `Polytope` objects, can be non-convex.

## Convex Polytopes

A `Polytope` is defined by passing the matrices `A` and `b` as `numpy` arrays:

```python
import numpy as np

A = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [-1.0, -0.0],
              [-0.0, -1.0]])

b = np.array([2.0, 1.0, 0.0, 0.0])

p = pc.Polytope(A, b)
```

In this particular case we created a `Polytope` which happens to be aligned with
the coordinate axes.  So we can instantiate the exact same `Polytope` by calling
the convenience function `box2poly` instead:

```python
p = pc.box2poly([[0, 2], [0, 1]])
```

To verify we got what we wanted `print(p)` shows:

```python
Single polytope 
 [[ 1.  0.] |    [[ 2.]
  [ 0.  1.] x <=  [ 1.]
  [-1. -0.] |     [ 0.]
  [-0. -1.]]|     [ 0.]]
```

We can check if the point with coordinates `[0.5, 0.5]` is in `Polytope` `p`
with the expression:

```python
[0.5, 0.5] in p
```

We can compare polytopes:

  - `p1 <= p2` is `True` iff `p1` is a subset of `p2`
  - `p1 == p2` iff `p1 <= p2` and `p2 <= p1`

Set operations between polytopes are available as methods (as well as
functions):

```python
p1.union(p2)
p1.diff(p2)
p1.intersect(p2)
```

Some additional operations are available:

```python
p1.project(dim)
p1.scale(10) # b := 10 * b
```

Various `Polytope` characteristics are accessible via attributes:

```python
p.dim # number of dimensions of ambient Euclidean space
p.volume # measure in ambient space
p.chebR # Chebyshev ball radius
p.chebXc # Chebyshev ball center
p.cheby
p.bounding_box
```

Several of these attributes are properties that compute the associated
attribute, if it has not been already computed.  The reason is that some
computations, e.g., volume or [Chebyshev
ball](https://en.wikipedia.org/wiki/Chebyshev_center) radius, require sampling
or solving an optimization problem, which can be computationally expensive.

Finally, the method `plot` does what it says on a
[matplotlib](http://matplotlib.org) figure and `text` can be used for placing
annotations at the `Polytope`'s Chebyshev center.  The `bounding_box` can be
used to set the correct axis limits to ensure the `Polytope` is visible in the
plot.

## Regions

A `Region` is a (possibly non-convex and even disconnected) container of (by
definition convex) `Polytope` objects.  The polytopes can be passed in an
[iterable](https://docs.python.org/2.7/glossary.html#term-iterable) during
instantiation:

```python
p1 = pc.box2poly([[0,2], [0,1]])
p2 = pc.box2poly([[2,3], [0,2]])
r = pc.Region([p1, p2])
```

The above results in a `Region` representing a non-convex polytope comprised of
two convex polytopes.  Iteration is over the `Region`'s polytopes:

```python
for polytope in r:
    print(polytope)
```

The `Region` itself can also be displayed with `print(r)`.  Polytopes in a
`Region` are ordered in a list, so `Region[i]` returns the `i`-th `Polytope` in
that list.  The number of polytopes in a `Region` is `len(r)`.  For a single
`Polytope` calling `len(p)` returns 0.

Addition and subtraction return new `Region` objects:

```python
r1 + r2 # set union
r1 - r2 # set difference
r1 & r2 # intersection
```

The other methods and attributes have names identical with those of `Polytope`.

## Functions

An incomplete list of additional functions besides those described above:

- `is_empty`
- `is_fulldim`
- `is_convex`
- `is_adjacent(r1, r2)`: enlarge both by little and check for intersection
- `is_interior(r0, r1)`: `True` iff `r1` enlarged by little is still a subset of
                         `r0`
- `reduce`: to remove redundant inequalities from the H-representation
- `separate`: divide a `Region` into its connected components
- `envelope`: defined by all "outer" inequalities
- `extreme`: to compute the extreme points of a bounded `Polytope`
- `qhull`: convex hull
