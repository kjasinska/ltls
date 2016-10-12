# A reprodicton of code for experiments done during the internship.

import argparse
import logging
from time import time

from sklearn.metrics import precision_score

from ltls import LTLS2_ML, LTLS
from utils.io import load_dataset, formated_time
from utils.log import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='LTLS training arguments.')
    parser.add_argument('path_train', type=str, help='training data')
    parser.add_argument('n_features', type=int, help='number of features')
    parser.add_argument('--path_validation', type=str, default=None, help='validation data')
    parser.add_argument('--log', type=str, default=None, help='log file')
    parser.add_argument('--policy', type=str, default='ranking', help='Class/label to path assignment policy.')
    parser.add_argument('--model_dir', type=str, default=None, help='A directory where to store the model')
    parser.add_argument('--it', type=int, default=1, help='Iterations')
    parser.add_argument('--l', type=float, default=0.0, help='L1 regularization constant')

    parser.add_argument('--multilabel', help='multilabel classification, default: False (multiclass)',
                        action='store_true')
    parser.add_argument('--es', help='Stop early', action='store_true')
    parser.add_argument('--validate', help='Test train and validation error after every iteration', action='store_true')
    args = parser.parse_args()
    print args
    return args.path_train, args.path_validation, args.log, args.policy, args.model_dir, args.it, \
           args.l, args.multilabel, args.es, args.validate, args.n_features


def _train(path_train, path_validation, policy, model_dir, it, l, multilabel, es, validate, n_features):
    ####################################################################################################################
    t0 = time()
    logging.info("Reading files {0} and {1}".format(path_train, path_validation))

    X, y, X_valid, y_valid, le = load_dataset(path_train, n_features, path_valid=path_validation, multilabel=multilabel)

    n_classes = len(le.get_classes())

    logging.info("Reading time {0}".format(time() - t0))
    ####################################################################################################################

    regularization = l != 0
    if multilabel:
        model = LTLS2_ML.new_model(loss='ranking_separation', iterations=it, stop_early=es, validate=validate,
                                   regularization=regularization, lambd=l, policy=policy)
    else:
        model = LTLS.new_model(iterations=it, stop_early=es, validate=validate, regularization=regularization, lambd=l,
                               policy=policy)
    logging.info("Training started: {0}".format(formated_time()))
    t0 = time()

    model.fit(X, y, n_classes, n_features, le.get_classes(), X_valid=X_valid, y_valid=y_valid)

    logging.info("Training ended: {0}".format(formated_time()))
    logging.info("Training time: {0}".format(time() - t0))

    t0 = time()

    Y_hat = model.predict(X)

    logging.info("Prediction time {0}".format(time() - t0))
    train_prec = precision_score(y, Y_hat, average='micro')
    logging.info("Train precision@1 {0}".format(train_prec))

    if X_valid is not None:
        t0 = time()

        Yvalid_hat = model.predict(X_valid)

        logging.info("Prediction time {0}".format(time() - t0))
        valid_prec = precision_score(y_valid, Yvalid_hat, average='micro')
        logging.info("Valid precision@1 {0}".format(valid_prec))

    ####################################################################################################################
    logging.info("Saving model to {0}".format(model_dir))
    model.dump(model_dir)


def train():
    path_train, path_validation, log, policy, model_dir, it, l, multilabel, es, validate, n_features = parse_args()
    setup_logger(log)
    _train(path_train, path_validation, policy, model_dir, it, l, multilabel, es, validate, n_features)


if __name__ == "__main__":
    train()
