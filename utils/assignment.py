# A reprodicton of code for experiments done during the internship.

from random import shuffle
from utils import graph
import numpy as np
from scipy import sparse

from sklearn.utils.graph import graph_laplacian
from sklearn.utils.arpack import eigsh


class AssignerEdgeByEdge():
    def __init__(self, adj, n_classes, X, Y, shuffled=False):
        self.X = X
        self.Y = Y
        self.adj = adj
        self.n_classes = n_classes
        self.shuffled = shuffled

    def assign(self):
        hs = graph.compute_here_to_sink(self.adj, self.n_classes)
        num_nodes = len(self.adj)
        classes = dict(zip(range(self.n_classes), [{} for i in range(self.n_classes)]))
        for node in range(num_nodes):
            for succ in self.adj[node]:
                classes[succ][node] = []

        classes[0][-1] = [range(self.n_classes)]
        if self.shuffled:
            shuffle(classes[0][-1][0])

        class_to_edge = dict(zip(range(self.n_classes), [[] for i in range(self.n_classes)]))

        for node in range(num_nodes):
            how_many_from_each_pred_to_succ = self.create_group_counts(node, hs)
            successors_classes = self.assign_classes_to_successors(how_many_from_each_pred_to_succ, classes[node])
            for succ in successors_classes:
                for group in successors_classes[succ]:
                    classes[succ][node].append(group)
                    for clas in group:
                        class_to_edge[clas].append((node, succ))

        classmap = {}
        pathmap = {}
        for clas in range(self.n_classes):
            class_to_edge[clas] = tuple(class_to_edge[clas])
            path = tuple(class_to_edge[clas])
            classmap[clas] = path
            pathmap[path] = clas
        return classmap, pathmap

    def assign_classes_to_successors(self, how_many_from_each_pred_to_succ, classes_from_predecesors):
        successors_classes = dict(
            zip(how_many_from_each_pred_to_succ.keys(),
                [[] for i in range(len(how_many_from_each_pred_to_succ.keys()))]))
        for pred in classes_from_predecesors:
            classes_for_each_succ = self.divide_into_grous(classes_from_predecesors[pred],
                                                           how_many_from_each_pred_to_succ)
            for succ in classes_for_each_succ:
                for group in classes_for_each_succ[succ]:
                    successors_classes[succ].append(group)
        return successors_classes

    def divide_into_grous(self, classes_from_predecesor, how_many_from_each_pred_to_succ):
        divided = {}
        for succ in how_many_from_each_pred_to_succ:
            divided[succ] = []
        for group in classes_from_predecesor:
            remaining_classes = group
            for succ in how_many_from_each_pred_to_succ:
                group_for_succ = []
                for i in range(how_many_from_each_pred_to_succ[succ]):
                    group_for_succ.append(self.yieldelem(remaining_classes))
                divided[succ].append(group_for_succ)
            assert len(remaining_classes) == 0
        return divided

    def create_group_counts(self, node, hs):
        groups = {}
        for succ in self.adj[node]:
            groups[succ] = hs[succ]
        return groups

    def yieldelem(self, s):
        ret = None
        for i in s:
            ret = i
            break
        assert (ret is not None)
        s.remove(ret)
        return ret


class AssignerSeriation():
    def __init__(self, remaining, n_classes, Y, edge_to_num):

        self.paths = list(remaining)
        self.n_classes = n_classes
        self.Y = Y
        self.edge_to_num = edge_to_num

    def path_edge_matrix(self, paths):
        n_edges = len(self.edge_to_num)
        G = np.zeros((self.n_classes, n_edges))
        path_index = dict(zip(range(self.n_classes), paths))
        for path, index in zip(paths, range(self.n_classes)):
            for edge in path:
                G[index, self.edge_to_num[edge]] = 1
        return G, path_index

    def seriation(self, A):
        n_components = 2
        eigen_tol = 0.00001
        if sparse.issparse(A):
            A = A.todense()
        np.fill_diagonal(A, 0)
        laplacian, dd = graph_laplacian(A, return_diag=True)
        laplacian *= -1
        lambdas, diffusion_map = eigsh(laplacian, k=n_components, sigma=1.0, which='LM', tol=eigen_tol)
        embedding = diffusion_map.T[n_components::-1]  # * dd
        sort_index = np.argsort(embedding[1])
        return sort_index

    def assign(self):

        G, path_index = self.path_edge_matrix(self.paths)
        H = np.dot(G, G.T)

        perm_paths = self.seriation(H)

        Y = sparse.csr_matrix(self.Y, dtype=np.float32)
        D = np.dot(Y.transpose(), Y)
        perm_classes = self.seriation(D)

        classmap = {}
        pathmap = {}

        for i in xrange(len(perm_classes)):
            classmap[perm_classes[i]] = path_index[perm_paths[i]]
            pathmap[path_index[perm_paths[i]]] = perm_classes[i]
        return classmap, pathmap
