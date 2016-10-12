# A reprodicton of code for experiments done during the internship.

import itertools
import random
import time

import numpy as np
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


class LabelEncoder2():
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        if self.multilabel:
            self.le = MultiLabelBinarizer(sparse_output=True)
        else:
            self.le = LabelEncoder()
        self.from_classes = False

    def fit(self, Y):
        self.le.fit(Y)
        self.from_classes = False

    def transform(self, Y):
        if self.from_classes:
            if self.multilabel:
                all_in_Y = set(itertools.chain.from_iterable(Y))
            else:
                all_in_Y = set()
                for el in np.unique(Y):
                    all_in_Y.add(el)
            self.new_classes = sorted(all_in_Y.difference(set(self.classes_)))
            self.num_in_training = len(self.classes_)
            self.num_new_in_test = len(self.new_classes)

            Y2 = []
            if self.multilabel:
                for yy in Y:
                    y2 = []
                    for y in yy:
                        if y in self.classes_:
                            y2.append(y)
                    Y2.append(y2)
            else:
                for y in Y:
                    if y in self.classes_:
                        Y2.append(y)
                    else:
                        # TODO
                        # random class? or remove the example? or _no_class_ marker?
                        Y2.append(random.choice(self.classes_))
            Y = Y2

            self.le.classes_ = self.classes_
        return self.le.transform(Y)

    def inverse_transform(self, Y):
        Y = self.le.inverse_transform(Y)
        return Y

    def set_classes(self, classes_):
        self.classes_ = classes_
        self.from_classes = True

    def get_classes(self):
        return self.le.classes_


def load_dataset(path_train, n_features, path_valid=None, path_test=None, multilabel=False, classes_=None):
    le = LabelEncoder2(multilabel=multilabel)
    if path_valid is None and path_test is None:  # TODO zero_based=True?
        X, Y = load_svmlight_file(path_train, dtype=np.float32, n_features=n_features, multilabel=multilabel)
        if classes_ is None:
            le.fit(Y)
            Y = le.transform(Y)
        else:
            le.set_classes(classes_)
            Y = le.transform(Y)
        return X, Y, None, None, le
    elif path_test is None:
        X, Y, Xvalid, Yvalid = load_svmlight_files((path_train, path_valid), dtype=np.float32,
                                                   n_features=n_features,
                                                   multilabel=multilabel)
        if classes_ is None:
            le.fit(np.concatenate((Y, Yvalid), axis=0))
            Y = le.transform(Y)
            Yvalid = le.transform(Yvalid)
        else:
            le.set_classes(classes_)
            Y = le.transform(Y)
            Yvalid = le.transform(Yvalid)
        return X, Y, Xvalid, Yvalid, le

    else:
        X, Y, Xvalid, Yvalid, Xtest, Ytest = load_svmlight_files((path_train, path_valid, path_test), dtype=np.float32,
                                                                 n_features=n_features,
                                                                 multilabel=multilabel)
        if classes_ is None:
            le.fit(np.concatenate((Y, Yvalid, Ytest), axis=0))
            Y = le.transform(Y)
            Yvalid = le.transform(Yvalid)
            Ytest = le.transform(Ytest)
        else:
            le.set_classes(classes_)
            Y = le.transform(Y)
            Yvalid = le.transform(Yvalid)
        return X, Y, Xvalid, Yvalid, Xtest, Ytest, le


def formated_time():
    return time.strftime("%d %b %Y %H:%M:%S", time.gmtime())
