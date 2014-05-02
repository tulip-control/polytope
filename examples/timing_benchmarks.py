"""
execution time measurements for polytope package
"""
import os
import numpy as np
import polytope as pc

if os.name is not 'posix':
    raise Exception('works only on POSIX operating systems')

# [0, 1] x [0, 1]
A0 = np.array([
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0]
])

b0 = np.array([[1.0, 0.0, 1.0, 0.0]])

# [0, 0.5] x [0, 0.5]
A1 = np.array([
    [0.0, 2.0],
    [0.0, -1.0],
    [2.0, 0.0],
    [-1.0, 0.0]
])

b1 = np.array([1.0, 0.0, 1.0, 0.0])

N = 10**4

print('starting timing measurements...')

# instance creation
start = os.times()[4]
for i in range(N):
    p0 = pc.Polytope(A0, b0)

end = os.times()[4]
print('instantiation: ' + str(end - start))

# intersection
p0 = pc.Polytope(A0, b0)
p1 = pc.Polytope(A1, b1)

start = os.times()[4]
for i in range(N):
    union = p0.intersect(p1)
end = os.times()[4]
print('intersection: ' + str(end - start))

start = os.times()[4]
for i in range(N):
    union = p0.union(p1)
end = os.times()[4]
print('union: ' + str(end - start))

start = os.times()[4]
for i in range(N):
    union = p0.diff(p1)
end = os.times()[4]
print('difference: ' + str(end - start))

print('end of timing measurements.')
