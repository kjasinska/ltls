# A reprodicton of code for experiments done during the internship.

from collections import defaultdict
import numpy as np
import math


def compute_here_to_sink(adj, n_classes):
    hs = {}
    num_nodes = len(adj) + 1
    sink = num_nodes - 1
    hs[sink] = 1

    for node in range(sink-1, -1, -1):
        hs[node] = 0
        for succ in adj[node]:
            hs[node] += hs[succ]
    assert n_classes == hs[0]
    return hs


def gengraph(k):
    b = list()
    kk = k
    while kk > 0:
        b.append(kk % 2)
        kk /= 2
    layers = len(b)
    source = 0
    curtop = 1
    curbot = 2
    sink = 2 * layers
    presink = sink - 1

    adj = defaultdict(list)
    adj[source].extend([curtop, curbot])
    adj[presink].append(sink)

    for i in xrange(layers - 1):
        nexttop = curtop + 2
        nextbot = curbot + 2
        if b[i] > 0:
            adj[curtop].append(sink)
        if nexttop <= presink:
            adj[curtop].append(nexttop)
            adj[curbot].append(nexttop)
        if nextbot < presink:
            adj[curtop].append(nextbot)
            adj[curbot].append(nextbot)
        curtop = nexttop
        curbot = nextbot

    paths = defaultdict(int)
    paths[0] = 1
    for i in xrange(sink):
        for j in adj[i]:
            paths[j] += paths[i]
    assert (paths[sink] == k)

    return adj


def count_edge_skips(num_edges, adj, num, branching_factor):
    skipped = np.ones((1, num_edges))
    sink = len(adj)
    for parent in adj:
        if len(adj[parent]) > branching_factor:
            skipped[0, num[(parent, sink)]] = math.ceil(sink / parent)
    return skipped


def enumerate_egdes(adj):
    k = 0
    num = dict()
    edg = dict()
    for i in adj:
        for j in adj[i]:
            edg[k] = (i, j)
            num[(i, j)] = k
            k += 1
    return k, num, edg


def genpathshelp(adj, prefix, paths):
    current = prefix[-1]
    sink = len(adj)
    for j in adj[current]:
        prefix.append(j)
        if j == sink:
            edges = zip(prefix[:-1], prefix[1:])
            paths.append(tuple(edges))
        else:
            genpathshelp(adj, prefix, paths)
        prefix.pop()
    return paths


def genpaths(adj):
    paths = list()
    prefix = [0]
    genpathshelp(adj, prefix, paths)
    path_to_index = {}
    for index in xrange(len(paths)):
        path_to_index[paths[index]] = index
    return paths, path_to_index
