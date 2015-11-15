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
Functions for plotting Partitions.
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import networkx as nx

from polytope import cheby_ball

# inline imports:
#
# import matplotlib as mpl
# from tulip.graphics import newax

def plot_partition(
    ppp, trans=None, ppp2trans=None, only_adjacent=False,
    ax=None, plot_numbers=True, color_seed=None
):
    """Plot partition with arrows from digraph.

    For filtering edges based on label use L{plot_ts_on_partition}.

    See Also
    ========
    L{abstract.prop2partition.PropPreservingPartition}, L{plot_trajectory}

    @type ppp: L{PropPreservingPartition}

    @param trans: Transition matrix. If used,
        then transitions in C{ppp} are shown with arrows.
        Otherwise C{ppp.adj} is plotted.

        To show C{ppp.adj}, pass: trans = True

    @param plot_numbers: If True,
        then annotate each Region center with its number.

    @param ax: axes where to plot

    @param color_seed: seed for reproducible random coloring

    @param ppp2trans: order mapping ppp indices to trans states
    @type ppp2trans: list of trans states
    """
    try:
        import matplotlib as mpl
        from tulip.graphics import newax
    except:
        logger.error('failed to import matplotlib')
        return

    # needs to be converted to adjacency matrix ?
    if isinstance(trans, nx.MultiDiGraph):
        if trans is not None and ppp2trans is None:
            msg = 'trans is a networkx MultiDiGraph, '
            msg += 'so ppp2trans required to define state order,\n'
            msg += 'used when converting the graph to an adjacency matrix.'
            raise Exception(msg)

        trans = nx.to_numpy_matrix(trans, nodelist=ppp2trans)
        trans = np.array(trans)

    l,u = ppp.domain.bounding_box
    arr_size = (u[0,0]-l[0,0])/50.0

    # new figure ?
    if ax is None:
        ax, fig = newax()

    # no trans given: use partition's
    if trans is True and ppp.adj is not None:
        ax.set_title('Adjacency from Partition')
        trans = ppp.adj
    elif trans is None:
        trans = 'none'
    else:
        ax.set_title('Adjacency from given Transitions')

    ax.set_xlim(l[0,0],u[0,0])
    ax.set_ylim(l[1,0],u[1,0])

    # repeatable coloring ?
    if color_seed is not None:
        prng = np.random.RandomState(color_seed)
    else:
        prng = np.random.RandomState()

    # plot polytope patches
    for i, reg in enumerate(ppp.regions):
        # select random color,
        # same color for all polytopes in each region
        col = prng.rand(3)

        # single polytope or region ?
        reg.plot(color=col, ax=ax)
        if plot_numbers:
            reg.text(str(i), ax, color='red')

    # not show trans ?
    if trans is 'none':
        mpl.pyplot.show()
        return ax

    # plot transition arrows between patches
    rows, cols = np.nonzero(trans)
    for i, j in zip(rows, cols):
        # mask non-adjacent cell transitions ?
        if only_adjacent:
            if ppp.adj[i, j] == 0:
                continue

        plot_transition_arrow(ppp.regions[i], ppp.regions[j], ax, arr_size)

    mpl.pyplot.show()

    return ax

def plot_transition_arrow(polyreg0, polyreg1, ax, arr_size=None):
    """Plot arrow starting from polyreg0 and ending at polyreg1.

    @type polyreg0: L{Polytope} or L{Region}
    @type polyreg1: L{Polytope} or L{Region}
    @param ax: axes where to plot

    @return: arrow object
    """
    try:
        from matplotlib import patches
    except:
        logger.error('failed to import matplotlib')
        return

    # brevity
    p0 = polyreg0
    p1 = polyreg1

    rc0, xc0 = cheby_ball(p0)
    rc1, xc1 = cheby_ball(p1)

    if np.sum(np.abs(xc1-xc0)) < 1e-7:
        return None

    if arr_size is None:
        l,u = polyreg1.bounding_box
        arr_size = (u[0,0]-l[0,0])/25.0

    #TODO: 3d
    x = xc0[0]
    y = xc0[1]
    dx = xc1[0] - xc0[0]
    dy = xc1[1] - xc0[1]
    arrow = patches.Arrow(
        float(x), float(y), float(dx), float(dy),
        width=arr_size, color='black'
    )
    ax.add_patch(arrow)

    return arrow
