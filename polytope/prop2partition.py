# Copyright (c) 2011-2014 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
"""
Proposition preserving partition module.
"""
import logging
logger = logging.getLogger(__name__)

import warnings

import numpy as np
from scipy import sparse as sp
import networkx as nx

import polytope as pc

_hl = 40 * '-'

def find_adjacent_regions(partition):
    """Return region pairs that are spatially adjacent.

    @type partition: iterable container of L{Region}

    @rtype: lil_matrix
    """
    n = len(partition)
    adj = sp.lil_matrix((n, n), dtype=np.int8)
    s = partition.regions

    for i, a in enumerate(s):
        adj[i, i] = 1

        for j, b in enumerate(s[0:i]):
            adj[i, j] = adj[j, i] = pc.is_adjacent(a, b)

    return adj

################################

class Partition(object):
    """Partition of a set.

    A C{Partition} is an iterable container of sets
    over C{Partition.set} and these must implement the methods:

        - union, __add__
        - difference
        - intersection
        - __le__

    so the builtin class C{set} can be used for discrete sets,
    or custom classes (e.g. polytopes) can be used for sets
    equipped with more structure.

    To utilize additional structure, see L{MetricPartition}.
    """
    def __init__(self, domain=None):
        """Partition over C{domain}.

        C{domain} is used to avoid conflicts with
        the python builtin set function.
        """
        self.set = domain

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def __getitem__(self, key):
        return self.regions[key]

    def is_partition(self):
        """Return True if Regions are pairwise disjoint and cover domain.
        """
        return self.is_cover() and self.are_disjoint()

    def is_cover(self):
        """Return True if Regions cover domain
        """
        union = pc.Region()
        for region in self.regions:
            union += region

        if not self.domain <= union:
            msg = 'partition does not cover domain.'
            logger.Error(msg)
            warnings.warn(msg)
            return False
        else:
            return True

    def are_disjoint(self, check_all=False, fname=None):
        """Return True if all Regions are disjoint.

        Print:

            - the offending Regions and their
            - their intersection (mean) volume ratio
            - their difference (mean) volume ratio

        Optionally save numbered figures of:

            - offending Regions
            - their intersection
            - their difference

        @param check_all: don't return when first offending regions found,
            continue and check all pairs
        @type check_all: bool

        @param fname: path prefix where to save the debugging figures
            By default no figures are saved.
        @type fname: str
        """
        logger.info('checking if PPP is a partition.')

        l,u = self.set.bounding_box
        ok = True
        for i, region in enumerate(self.regions):
            for j, other in enumerate(self.regions[0:i]):
                if pc.is_fulldim(region.intersect(other) ):
                    msg = 'PPP is not a partition, regions: '
                    msg += str(i) + ' and: ' + str(j)
                    msg += ' intersect each other.\n'
                    msg += 'Offending regions are:\n' + 10*'-' + '\n'
                    msg += str(region) + 10*'-' + '\n'
                    msg += str(other) + 10*'-' + '\n'

                    isect = region.intersect(other)
                    diff = region.diff(other)

                    mean_volume = (region.volume + other.volume) /2.0

                    overlap = 100 * isect.volume / mean_volume
                    non_overlap = 100 * diff.volume / mean_volume

                    msg += '|cap| = ' + str(overlap) + ' %\n'
                    msg += '|diff| = ' + str(non_overlap) + '\n'

                    logger.error(msg)

                    if fname:
                        print('saving')
                        fname1 = fname + 'region' + str(i) + '.pdf'
                        fname2 = fname + 'region' + str(j) + '.pdf'
                        fname3 = fname + 'isect_' + str(i) + '_' + str(j) + '.pdf'
                        fname4 = fname + 'diff_' + str(i) + '_' + str(j) + '.pdf'

                        _save_region_plot(region, fname1, l, u)
                        _save_region_plot(other, fname2, l, u)
                        _save_region_plot(isect, fname3, l, u)
                        _save_region_plot(diff, fname4, l, u)

                    ok = False
                    if not check_all:
                        break
        return ok

    def refines(self, other):
        """Return True if each element is a subset of other.

        @type other: PropPreservingPartition
        """
        for small in self:
            found_superset = False
            for big in other:
                if small <= big:
                    found_superset = True
                    break
            if not found_superset:
                return False
        return True

    def preserves(self, other):
        """Return True if it refines closure of C{other} under complement.

        Closure under complement is the union of C{other}
        with the collection of complements of its elements.

        This method checks the annotation of elements in C{self}
        with elements fro C{other}.
        """
        for item in self._elements:
            # item subset of these sets
            for superset in item.supersets:
                if not item <= superset:
                    return False

            # item subset of the complements of these sets
            for other_set in set(other).difference(item.supersets):
                if item.intersect(other_set):
                    return False
        return True

class MetricPartition(Partition, nx.Graph):
    """Partition of a metric space.

    Includes adjacency information which abstracts
    the topology induced by the metric.

    Two subsets in the partition are called adjacent
    if the intersection of their closure is non-empty.

    If the space is also a measure space,
    then volume information is used for diagnostic purposes.
    """
    def compute_adj(self):
        """Update the adjacency matrix by checking all region pairs.

        Uses L{polytope.is_adjacent}.
        """
        n = len(self.regions)
        adj = sp.lil_matrix((n, n))

        logger.info('computing adjacency from scratch...')
        for i, region0 in enumerate(self.regions):
            for j, region1 in enumerate(self.regions):
                if i == j:
                    adj[i, j] = 1
                    continue

                if pc.is_adjacent(region0, region1):
                    adj[i, j] = 1
                    adj[j, i] = 1

                    logger.info('regions: ' + str(i) + ', ' +
                                 str(j) + ', are adjacent.')
        logger.info('...done computing adjacency.')

        # check previous one to unmask errors
        if self.adj is not None:
            logger.info('checking previous adjacency...')

            ok = True
            row, col = adj.nonzero()

            for i, j in zip(row, col):
                assert(adj[i, j])
                if adj[i, j] != self.adj[i, j]:
                    ok = False

                    msg = 'PPP adjacency matrix is incomplete, '
                    msg += 'missing: (' + str(i) + ', ' + str(j) + ')'
                    logger.error(msg)

            row, col = self.adj.nonzero()

            for i, j in zip(row, col):
                assert(self.adj[i, j])
                if adj[i, j] != self.adj[i, j]:
                    ok = False

                    msg = 'PPP adjacency matrix is incorrect, '
                    msg += 'has 1 at: (' + str(i) + ', ' + str(j) + ')'
                    logger.error(msg)

            if not ok:
                logging.error('PPP had incorrect adjacency matrix.')

            logger.info('done checking previous adjacency.')
        else:
            ok = True
            logger.info('no previous adjacency found: ' +
                        'skip verification.')

        # update adjacency
        self.adj = adj

        return ok

def _save_region_plot(region, fname, l, u):
    ax = region.plot()
    ax.set_xlim(l[0,0], u[0,0])
    ax.set_ylim(l[1,0], u[1,0])
    ax.figure.savefig(fname)
