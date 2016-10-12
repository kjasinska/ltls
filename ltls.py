# A reprodicton of code for experiments done during the internship.

import logging
import math
import os
import pickle
from itertools import izip
import numpy as np
from scipy import sparse
from sklearn.metrics import precision_score

from utils import assignment, graph, ranking, io


class LTLS:
    filename_model = "model"
    filename_path_to_class_map = "path_to_class"
    filename_w = "w"

    POLICY_RANDOM = "random"
    POLICY_ORDER = "order"
    POLICY_RANKING = "ranking"

    def __init__(self, iterations=5, stop_early=False, validate=False, regularization=False,
                 lambd=0, policy="ranking", w=None, class_to_path_map=None, path_to_class_map=None, classes_=None,
                 n_classes=None, n_features=None, loaded=False, multiplier=1):

        self.n_classes = n_classes
        self.n_features = n_features
        self.iterations = iterations
        self.stop_early = stop_early
        self.stop_early_c = 3
        self.regularization = regularization
        self._lambd = lambd
        self.policy = policy
        self.lambd = 0
        self.classes_ = classes_
        self.validate = validate
        self.in_training = False
        self.u = None
        self.margin = 1
        self.xu = None
        self.max_updates = 10
        self.adj = None
        self.num_edges = None
        self.edge_to_num = None
        self.num_to_edge = None
        self.paths = None
        self.path_to_index = None
        self.skipped = None
        self.iterate_dataset = None
        self.remaining = set()
        self.iterate_dataset = self.iterate_original
        self.multiplier = multiplier
        if loaded:
            self.create_graph(n_classes)
            self.wa = None
            self.w = w
            self.class_to_path_map = class_to_path_map
            self.path_to_class_map = path_to_class_map
        else:
            self.class_to_path_map = dict()
            self.path_to_class_map = dict()

    @classmethod
    def new_model(cls, iterations=5, stop_early=False, validate=False, regularization=False,
                  lambd=0, policy="ranking", multiplier=1):
        return cls(iterations=iterations, stop_early=stop_early, validate=validate, regularization=regularization,
                   lambd=lambd, policy=policy, multiplier=multiplier)

    def create_graph(self, n_classes):
        self.adj = graph.gengraph(n_classes)
        self.num_edges, self.edge_to_num, self.num_to_edge = graph.enumerate_egdes(self.adj)
        self.paths, self.path_to_index = graph.genpaths(self.adj)
        self.skipped = graph.count_edge_skips(self.num_edges, self.adj, self.edge_to_num, 2)

    def iterate_original(self, X, Y=None):
        if Y is None:
            for i in range(X.shape[0]):
                yield X[i, :]
        else:
            for i in range(X.shape[0]):
                yield (X[i, :], Y[i])

    def create_ranking(self, edge_weight, k):
        return ranking.create_ranking3(edge_weight, k, self.adj, self.edge_to_num)

    def yieldelem(self, s):
        ret = None
        for i in s:
            ret = i
            break
        assert (ret is not None)
        s.remove(ret)
        return ret

    def get_path_weight(self, edge_weight, path):
        weight = 0
        for e in path:
            weight += edge_weight[0, self.edge_to_num[e]]
        return weight

    def get_label_path_weight(self, edge_weight, y):
        truepath = self.class_to_path_map[y]
        weight = self.get_path_weight(edge_weight, truepath)
        return truepath, weight

    def get_pred_true_paths_for_loss(self, x, y):
        edge_weight = self.evaluate_model(x, self.w)
        if self.class_to_path_map.has_key(y):
            truepath, truepath_weight = self.get_label_path_weight(edge_weight, y)
            ranking = self.create_ranking(edge_weight, 2)  # multiclass!
        else:
            ranking = self.create_ranking(edge_weight,
                                          min(len(self.class_to_path_map.keys()) + 2, self.ranking_max_length))
            truepath, truepath_weight = self.assign_first_free_path_in_ranking_or_yield(ranking, y, edge_weight)
        predpath, predpath_weight = self.get_highest_neg_path(ranking, truepath)
        return predpath, truepath, predpath_weight, truepath_weight

    def assign_first_free_path_in_ranking_or_yield(self, logarithmic_ranking, y, edge_weight):
        for weight, path in logarithmic_ranking:
            path = tuple(path)
            if path not in self.path_to_class_map:
                self.class_to_path_map[y] = path
                self.path_to_class_map[path] = y
                self.remaining.remove(path)
                return path, weight
        return self.assigin_yield(edge_weight, y)

    def assigin_yield(self, edge_weight, y):
        path = self.yieldelem(self.remaining)
        self.class_to_path_map[y] = path
        self.path_to_class_map[path] = y
        weight = self.get_path_weight(edge_weight, path)
        return path, weight

    def get_highest_neg_path(self, ranking, truepath):  # multiclass case
        highest_neg_rank = 0
        if truepath == ranking[0][1]:
            highest_neg_rank = 1
        highest_neg_path_weight = ranking[highest_neg_rank][0]
        highest_neg_path = tuple(ranking[highest_neg_rank][1])
        return highest_neg_path, highest_neg_path_weight

    def initialize_model(self):
        self.create_graph(self.n_classes)
        self.up = np.ndarray((1, self.num_edges))
        self.wa = np.zeros((self.n_features, self.num_edges), dtype=np.float32)
        self.w = np.zeros((self.n_features, self.num_edges), dtype=np.float32)
        self.ranking_max_length = self.multiplier * int(math.ceil(math.log(self.n_classes, 2)))
        self.u = np.zeros((1, self.num_edges))
        self.xu = np.zeros((1, self.num_edges))

    def fit(self, X, y, n_classes, n_features, classes_, X_valid=None, y_valid=None):
        self.in_training = True
        self.n_classes = n_classes
        self.n_features = n_features
        self.classes_ = classes_
        self.initialize_model()

        c = 1
        worse_count = 0
        best_valid_precision = 0
        best_w = np.zeros_like(self.w)
        best_iteration = 0

        if self.policy == self.POLICY_RANDOM:
            self.assign_before_training(X, y, shuffled=True)
            self.remaining = set()
        elif self.policy == self.POLICY_ORDER:
            self.assign_before_training(X, y, shuffled=False)
            self.remaining = set()
        else:
            paths, _ = graph.genpaths(self.adj)
            self.remaining = set(paths)

        for t in xrange(self.iterations):
            logging.info("Iteration {0}: {1}".format(t, io.formated_time()))
            if t > 0:
                self.lambd = self._lambd

            for x, yy in self.iterate_dataset(X, y):
                predpath, truepath, predpath_weight, truepath_weight = self.get_pred_true_paths_for_loss(x, yy)
                if predpath_weight + self.margin >= truepath_weight:
                    self.update_model_row(c, predpath, truepath, x)
                c += 1

            if len(self.remaining) > 0:
                self.assign_remaining_to_unseen_classes()

            if self.validate or self.stop_early:
                wtemp = self.w - self.wa / c

                train_prec = self.evaluate(wtemp, X, y)
                logging.info("Train precision@1 {0}".format(train_prec))

                valid_prec = None
                if X_valid is not None:
                    valid_prec = self.evaluate(wtemp, X_valid, y_valid)
                    logging.info("Valid precision@1 {0}".format(valid_prec))

                if self.stop_early and X_valid is not None:
                    best_iteration, best_valid_precision, worse_count = self.check_early_stopping(t, best_iteration,
                                                                                                  best_valid_precision,
                                                                                                  best_w, valid_prec,
                                                                                                  worse_count, wtemp)
                    if worse_count >= self.stop_early_c:
                        logging.info("Stopping after {0} iterations".format(t))
                        break

        if self.stop_early:
            logging.info("Best number of iterations {0}".format(best_iteration + 1))
            logging.info("Best validation precision {0}".format(best_valid_precision))
            self.w = best_w
        else:
            self.w = self.w - self.wa / c

    def assign_remaining_to_unseen_classes(self):
        unassigned_paths = list(self.remaining)
        unseen_classes = list(set(range(self.n_classes)).difference(set(self.class_to_path_map.keys())))
        assert (len(unassigned_paths) == len(unseen_classes))
        for i in range(len(unassigned_paths)):
            self.class_to_path_map[unseen_classes[i]] = unassigned_paths[i]
            self.path_to_class_map[unassigned_paths[i]] = unseen_classes[i]
        self.remaining = set()

    def assign_before_training(self, X, Y, shuffled=False):
        assign = assignment.AssignerEdgeByEdge(self.adj, self.n_classes, X, Y, shuffled=shuffled)
        self.class_to_path_map, self.path_to_class_map = assign.assign()
        self.remaining = set()

    def check_early_stopping(self, t, prev_it, prev_valid_prec, prev_w, valid_prec, worse_counter, wtemp):
        if valid_prec > prev_valid_prec:
            prev_valid_prec = valid_prec
            prev_w[:] = wtemp
            prev_it = t
            worse_counter = 0
        else:
            worse_counter += 1
        return prev_it, prev_valid_prec, worse_counter

    def update_model_row(self, c, predpath, truepath, x):
        self.u.fill(0)
        for e in predpath:
            self.u[0, self.edge_to_num[e]] -= 1
        for e in truepath:
            self.u[0, self.edge_to_num[e]] += 1

        for feature, value in izip(x.indices, x.data):
            self.xu[:] = self.u
            self.xu *= value
            self.w[feature, :] = self.w[feature, :] + self.xu
            self.wa[feature, :] = self.wa[feature, :] + c * self.xu

    def evaluate(self, w, X, Y):
        Yhat = self._predict(X, w)
        return precision_score(Y, Yhat, average='micro')

    def evaluate_model(self, x, w):
        if not self.regularization or self.lambd == 0:
            edge_weight = x.dot(w)
            edge_weight = np.multiply(edge_weight, self.skipped)
        else:
            edge_weight = np.zeros((1, self.num_edges))
            for idx, value in izip(x.indices, x.data):
                # edge_weight = np.add(edge_weight, np.multiply(value, np.multiply(np.maximum(np.subtract(np.abs(w[idx, :]), self.lambd), 0), np.sign(w[idx, :]))))
                for edge in xrange(self.num_edges):
                    if w[idx, edge] > self.lambd:
                        edge_weight[0, edge] += value * (w[idx, edge] - self.lambd)
                    elif w[idx, edge] < -self.lambd:
                        edge_weight[0, edge] += value * (w[idx, edge] + self.lambd)
        return edge_weight

    def predict_topk(self, X, k=5):
        Yhat = np.zeros((X.shape[0], k))
        i = 0
        for x in self.iterate_dataset(X):
            edge_weight = self.evaluate_model(x, self.w)
            ranking = self.create_ranking(edge_weight, k)
            rank = 0
            for path in ranking:
                Yhat[i, rank] = self.path_to_class_map[path[1]]
                rank += 1
            i += 1
        return Yhat

    def predict_topk(self, X, k=5):
        Yhat = np.zeros((X.shape[0], k))
        i = 0
        for x in self.iterate_dataset(X):
            edge_weight = self.evaluate_model(x, self.w)
            ranking = self.create_ranking(edge_weight, k)
            rank = 0
            for path in ranking:
                Yhat[i, rank] = self.path_to_class_map[path[1]]
                rank += 1
            i += 1
        return Yhat

    def _predict_one(self, x, w):
        edge_weight = self.evaluate_model(x, w)
        source = 0
        sink = len(self.adj)
        nodes = sink + 1
        costto = [float('-inf')] * nodes
        arrive = [None] * nodes
        costto[source] = 0
        for i in xrange(sink):
            for j in self.adj[i]:
                ew = edge_weight[0, self.edge_to_num[(i, j)]]
                newcost = costto[i] + ew
                if newcost > costto[j]:
                    costto[j] = newcost
                    arrive[j] = i

        path = [sink]
        current = sink
        while current != source:
            current = arrive[current]
            path.append(current)
        path.reverse()
        edges = zip(path[:-1], path[1:])
        return tuple(edges)

    def _predict(self, X, w):
        Yhat = [0] * X.shape[0]
        i = 0
        for x in self.iterate_dataset(X):
            Yhat[i] = self.path_to_class_map[self._predict_one(x, w)]
            i += 1
        return Yhat

    def predict(self, X):
        return self._predict(X, self.w)

    def dump(self, dirname):
        try:
            os.mkdir(dirname)
        except:
            logging.info("Model dir already existed. Overwriting.")
        with open(os.path.join(dirname, self.filename_model), 'w') as f:
            f.write("{0}\n".format(self.n_features))
            f.write("{0}\n".format(self.n_classes))
            for clas in self.classes_:
                f.write("{0} ".format(int(clas)))
            f.write("\n")
            f.write("{0}\n".format(self.lambd))
        pickle.dump(self.path_to_class_map, open(os.path.join(dirname, self.filename_path_to_class_map), "wb"))
        np.save(os.path.join(dirname, self.filename_w), self.w)

    @classmethod
    def load(cls, dirname):
        with open(os.path.join(dirname, cls.filename_model), 'r') as f:
            n_features = int(f.readline().strip())
            n_classes = int(f.readline().strip())
            classes_ = [int(clas) for clas in f.readline().strip().split()]
            lambd = float(f.readline().strip())
        path_to_class_map = pickle.load(open(os.path.join(dirname, cls.filename_path_to_class_map), "rb"))
        class_to_path_map = dict()
        for path in path_to_class_map:
            class_to_path_map[path_to_class_map[path]] = path

        w = np.load(os.path.join(dirname, cls.filename_w + ".npy"))
        regularization = lambd != 0
        return cls(n_classes=n_classes, n_features=n_features, w=w, class_to_path_map=class_to_path_map,
                   path_to_class_map=path_to_class_map, classes_=classes_, lambd=lambd, loaded=True,
                   regularization=regularization)


class LTLS2_ML(LTLS):
    LOSS_RANKING_SEPARATION_ALL_POSITIVES = 'ranking_separation_all_positives'
    LOSS_RANKING_SEPARATION = 'ranking_separation'
    POLICY_SERIATION = 'seriation'

    def __init__(self, loss='ranking_separation', iterations=5, stop_early=False, validate=False, regularization=False,
                 lambd=0, policy="ranking", w=None, class_to_path_map=None, path_to_class_map=None, classes_=None,
                 n_classes=None, n_features=None, loaded=False, multiplier=1):

        LTLS.__init__(self, iterations=iterations, stop_early=stop_early, validate=validate,
                      regularization=regularization, lambd=lambd, policy=policy, w=w,
                      class_to_path_map=class_to_path_map, path_to_class_map=path_to_class_map,
                      classes_=classes_, n_classes=n_classes, n_features=n_features, loaded=loaded,
                      multiplier=multiplier)
        self.loss = loss

    @classmethod
    def new_model(cls, loss='ranking_separation', iterations=5, stop_early=False, validate=False, regularization=False,
                  lambd=0, policy='ranking', multiplier=1):
        return cls(loss=loss, iterations=iterations, stop_early=stop_early, validate=validate,
                   regularization=regularization,
                   lambd=lambd, policy=policy, multiplier=multiplier)

    def create_ranking(self, edge_weight, k):
        return ranking.create_ranking1(edge_weight, k, self.adj, self.edge_to_num)

    def iterate_original(self, X, Y=None):
        if Y is None:
            for i in range(X.shape[0]):
                yield X[i, :]
        else:
            for i in range(X.shape[0]):
                yield (X[i, :], Y[i].indices)

    def evaluate_model(self, x, w):
        edge_weight = x.dot(w)
        edge_weight = np.multiply(edge_weight, self.skipped)
        return edge_weight

    def get_highest_negative_score(self, edge_weight, y):
        ranking = self.create_ranking(edge_weight, len(y) + 1)
        for weight, path in ranking:
            path = tuple(path)
            if path not in self.path_to_class_map:
                return path, weight
            label = self.path_to_class_map[path]
            if label not in y:
                return path, weight

    def get_all_positive_scores(self, edge_weight, y):
        scores = []
        for label in y:
            path, score = self.get_label_path_weight(edge_weight, label)
            scores.append((path, score))
        return scores

    def get_unseen_labels(self, y):
        unseen = set(y).difference(self.classes_seen)
        for c in unseen:
            self.classes_seen.add(c)
        return unseen

    def assign_first_free_path_in_ranking_or_yield(self, logarithmic_ranking, y, edge_weight):
        remaining_before = len(self.remaining)
        for weight, path in logarithmic_ranking:
            path = tuple(path)
            if path not in self.path_to_class_map:
                self.class_to_path_map[y] = path
                self.path_to_class_map[path] = y
                self.remaining.remove(path)
                assert remaining_before > len(self.remaining)
                return path, weight
        return self.assigin_yield(edge_weight, y)

    def get_lowest_positive_score(self, edge_weight, y):
        if self.policy == self.POLICY_RANKING:
            unseen = self.get_unseen_labels(y)
            if len(unseen) != 0:
                ranking = self.create_ranking(edge_weight, min(len(self.class_to_path_map.keys()) + len(unseen),
                                                               self.ranking_max_length))
                for label in unseen:
                    _, _ = self.assign_first_free_path_in_ranking_or_yield(ranking, label, edge_weight)

        lowest_positive_score = float('+inf')
        lowest_score_path = None
        for label in y:
            # print label
            path, score = self.get_label_path_weight(edge_weight, label)
            if score < lowest_positive_score:
                lowest_positive_score = score
                lowest_score_path = path

        return lowest_positive_score, lowest_score_path

    def get_highest_positive_score(self, edge_weight, y):
        highest_positive_score = float('-inf')
        for label in y:
            # print label
            path, score = self.get_label_path_weight(edge_weight, label)
            if score > highest_positive_score:
                highest_positive_score = score
        return highest_positive_score

    # @profile
    def fit(self, X, y, n_classes, n_features, classes_, X_valid=None, y_valid=None):
        self.in_training = True
        self.n_classes = n_classes
        self.n_features = n_features
        self.classes_ = classes_
        self.classes_seen = set()
        self.initialize_model()
        c = 1
        worse_count = 0
        best_valid_precision = 0
        best_w = np.zeros_like(self.w)
        best_iteration = 0

        if self.policy == self.POLICY_RANDOM:
            self.assign_before_training(X, y, shuffled=True)
            self.remaining = set()
        elif self.policy == self.POLICY_ORDER:
            self.assign_before_training(X, y, shuffled=False)
            self.remaining = set()
        elif self.policy == self.POLICY_SERIATION:
            paths, _ = graph.genpaths(self.adj)
            self.remaining = set(paths)
            self.assign_before_training_seriation(y)
            self.remaining = set()
        else:
            self.seen_classes = set()
            paths, _ = graph.genpaths(self.adj)
            self.remaining = set(paths)

        for t in xrange(self.iterations):
            logging.info("Iteration {0}: {1}".format(t, io.formated_time()))
            for x, yy in self.iterate_dataset(X, y):
                edge_weight = self.evaluate_model(x, self.w)

                if self.loss == self.LOSS_RANKING_SEPARATION:
                    lowest_positive_score, lowest_score_path = self.get_lowest_positive_score(edge_weight, yy)
                    highest_score_negative_path, highest_negative_score = self.get_highest_negative_score(edge_weight,
                                                                                                          yy)
                    if highest_negative_score + self.margin > lowest_positive_score:
                        self.update_model_row(c, highest_score_negative_path, lowest_score_path, x)

                elif self.loss == self.LOSS_RANKING_SEPARATION_ALL_POSITIVES:
                    positive_update = []
                    highest_score_negative_path, highest_negative_score = self.get_highest_negative_score(edge_weight,
                                                                                                          yy)
                    scores = self.get_all_positive_scores(edge_weight, yy)
                    for path, score in scores:
                        if highest_negative_score + self.margin > score:
                            positive_update.append(path)
                    if len(positive_update) > 0:
                        self.update_model_row_multi_positive(c, positive_update, highest_score_negative_path, x)

                c += 1

            if len(self.remaining) > 0:
                self.assign_remaining_to_unseen_classes()

            if self.validate or self.stop_early:
                self.in_training = False
                wtemp = self.w - self.wa / c

                train_prec = self.evaluate(wtemp, X, y)
                logging.info("Train precision@1 {0}".format(train_prec))

                valid_prec = None
                if X_valid is not None:
                    valid_prec = self.evaluate(wtemp, X_valid, y_valid)
                    logging.info("Valid precision@1 {0}".format(valid_prec))

                self.in_training = True

                if self.stop_early and X_valid is not None:
                    best_iteration, best_valid_precision, worse_count = self.check_early_stopping(t, best_iteration,
                                                                                                  best_valid_precision,
                                                                                                  best_w, valid_prec,
                                                                                                  worse_count, wtemp)
                    if worse_count >= self.stop_early_c:
                        logging.info("Stopping after {0} iterations".format(t))
                        break

        if self.stop_early:
            logging.info("Best number of iterations {0}".format(best_iteration + 1))
            logging.info("Best validation precision {0}".format(best_valid_precision))
            self.w = best_w
        else:
            self.w = self.w - self.wa / c

    def assign_before_training_seriation(self, Y):
        assign = assignment.AssignerSeriation(self.remaining, self.n_classes, Y, self.edge_to_num)
        self.class_to_path_map, self.path_to_class_map = assign.assign()
        self.remaining = set()

    def predict_topk(self, X, k=5):
        row = [[] for i in xrange(k)]
        col = [[] for i in xrange(k)]
        i = 0
        for x in self.iterate_dataset(X):
            edge_weight = self.evaluate_model(x, self.w)
            ranking = self.create_ranking(edge_weight, k)
            rank = 0
            for path in ranking:
                yhat = self.path_to_class_map[path[1]]
                col[rank].append(yhat)
                row[rank].append(i)
                rank += 1
            i += 1
        Yhats = []
        for i in xrange(k):
            row1 = np.array(row[i])
            col1 = np.array(col[i])
            data1 = np.array([1 for i in xrange(len(row[i]))])
            Yhat = sparse.csr_matrix((data1, (row1, col1)), shape=(X.shape[0], self.n_classes))
            Yhats.append(Yhat)
        return Yhats

    def _predict(self, X, w):
        row = []
        col = []
        i = 0
        for x in self.iterate_dataset(X):
            yhat = self.path_to_class_map[self._predict_one(x, w)]
            col.append(yhat)
            row.append(i)  # top1 prediction!
            i += 1
        row = np.array(row)
        col = np.array(col)
        data = np.array([1 for i in xrange(len(row))])

        Yhat = sparse.csr_matrix((data, (row, col)), shape=(X.shape[0], self.n_classes))
        return Yhat

    def update_model_row_multi_positive(self, c, positive_updates, negative_path, x):
        self.u.fill(0)
        for path in positive_updates:
            for e in path:
                self.u[0, self.edge_to_num[e]] += 1

        for e in negative_path:
            self.u[0, self.edge_to_num[e]] -= len(positive_updates)

        for feature, value in izip(x.indices, x.data):
            self.xu[:] = self.u
            self.xu *= value
            self.w[feature, :] = self.w[feature, :] + self.xu
            self.wa[feature, :] = self.wa[feature, :] + c * self.xu
