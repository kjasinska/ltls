# A reprodicton of code for experiments done during the internship.

import heapq


# @profile
def merge_rankings(temp_ranking, ranking_parent, ew_parent_current, parent, current, k):
    new_ranking = []
    iterator_temp = 0
    iterator_parent = 0
    WEIGHT = 0
    PATH = 1
    while iterator_temp < len(temp_ranking) or iterator_parent < len(ranking_parent):
        # merge to new_ranking the rest of one of the rankings, since the second is already empty
        if iterator_temp >= len(temp_ranking):
            while iterator_parent < len(ranking_parent):
                new_path = list(ranking_parent[iterator_parent][PATH])
                new_path.append((parent, current))
                new_ranking.append((ranking_parent[iterator_parent][WEIGHT] + ew_parent_current, new_path))
                iterator_parent += 1
            break
        if iterator_parent >= len(ranking_parent):
            while iterator_temp < len(temp_ranking):
                new_ranking.append(temp_ranking[iterator_temp])
                iterator_temp += 1
            break
        # zip rankings
        if temp_ranking[iterator_temp][WEIGHT] > ranking_parent[iterator_parent][WEIGHT] + ew_parent_current:
            new_ranking.append(temp_ranking[iterator_temp])
            iterator_temp += 1
        elif temp_ranking[iterator_temp][WEIGHT] < ranking_parent[iterator_parent][WEIGHT] + ew_parent_current:
            new_path = list(ranking_parent[iterator_parent][PATH])
            new_path.append((parent, current))
            new_ranking.append((ranking_parent[iterator_parent][WEIGHT] + ew_parent_current, new_path))
            iterator_parent += 1
        else:
            new_ranking.append(temp_ranking[iterator_temp])
            iterator_temp += 1
            new_path = list(ranking_parent[iterator_parent][PATH])
            new_path.append((parent, current))
            new_ranking.append((ranking_parent[iterator_parent][WEIGHT] + ew_parent_current, new_path))
            iterator_parent += 1
        # cut-off, only k top elements required
        if len(new_ranking) >= k:
            break
    return new_ranking


# @profile
def create_ranking1(edge_weight, k, adj, num):
    sink = len(adj)
    rankings = {}
    rankings[0] = [(0, [])]  # [sic]
    to_merge = {}
    to_merge[0] = set()
    to_remove = {}
    for current in xrange(sink + 1):
        to_remove[current] = set()
        if current < sink:
            for child in adj[current]:
                if child not in to_merge.keys():
                    to_merge[child] = set()
                to_merge[child].add(current)
                to_remove[current].add(child)
        # create current node ranking:
        if current > 0:
            temp_ranking = []
            for parent in to_merge[current]:
                ew = edge_weight[0, num[(parent, current)]]
                temp_ranking = merge_rankings(temp_ranking, rankings[parent], ew, parent, current, k)
            rankings[current] = temp_ranking
            # remove current id from to_remove, if len(to_remove[x]) == 0 remove x
            for parent in to_merge[current]:
                to_remove[parent].remove(current)
                if len(to_remove[parent]) == 0:
                    # to_remove.pop(parent) # not really necessary
                    rankings.pop(parent)
    sink_ranking = []
    for weight, path in rankings[sink]:
        sink_ranking.append((weight, tuple(path)))
    return sink_ranking


# TODO remove this additional call
# @profile
def create_ranking2(edge_weight, k, adj, num):
    sink = len(adj)
    heaps = [[] for i in xrange(sink + 1)]
    heaps[0] = [(0, [])]

    for current in xrange(sink):
        for child in adj[current]:
            for length, path in heaps[current]:
                new_path = list(path)
                new_path.append(current)
                # this can be done better using this heapreplace
                ew = edge_weight[0, num[(current, child)]]
                heapq.heappush(heaps[child], (length + ew, new_path))
                heaps[child] = heapq.nlargest(k, heaps[child])
                # TODO what with equal lenght paths?
    # result: heaps[sink]
    return [(length, tuple(zip(nodes, nodes[1:] + [sink]))) for length, nodes in heaps[sink]]


def create_ranking3(edge_weight, k, adj, num):
    sink = len(adj)
    EMPTY = -2
    ROOT = -1
    MIN_LENGTH = float('-inf')
    # heaps = [[(0, EMPTY, 0) for j in range(k)] for i in xrange(sink + 1)]
    heaps = [[(MIN_LENGTH, EMPTY, 0) for j in range(k + 1)] for i in xrange(sink + 1)]
    heaps[0][0] = (0, ROOT, 0)
    # forward
    for current in xrange(sink):
        new_rank = 0
        for length, parent, rank in heaps[current]:
            if parent != EMPTY:
                for child in adj[current]:
                    ew = edge_weight[0, num[(current, child)]]
                    new_length = length + ew
                    # heapq.heapreplace(heaps[child], (new_length, current, new_rank))
                    heapq.heappush(heaps[child], (new_length, current, new_rank))
                    heaps[child] = heapq.nlargest(k, heaps[child])
            new_rank += 1
    # backward
    ranking = []
    for rank in xrange(k):
        path = []
        current = sink
        current_rank = rank
        while current != ROOT:
            path.append(current)
            _, current, current_rank = heaps[current][current_rank]
        length, _, _ = heaps[sink][rank]
        path = list(reversed(path))
        path = tuple(zip(path[:-1], path[1:]))
        ranking.append((length, path))
    return ranking


