# A reprodicton of code for experiments done during the internship.

import logging
from time import time

from sklearn.metrics import precision_score

from ltls import LTLS, LTLS2_ML
from utils.io import formated_time, load_dataset


def _train(path_train, path_validation, path_test, policy, model_dir, it, lambd, multilabel, es, validate, n_features,
           multplier):
    ####################################################################################################################
    t0 = time()
    logging.info("Reading files {0}, {1} and {2}".format(path_train, path_validation, path_test))

    X, y, X_valid, y_valid, X_test, y_test, le = load_dataset(path_train, n_features, path_valid=path_validation,
                                                              path_test=path_test, multilabel=multilabel)

    n_classes = len(le.get_classes())

    logging.info("Reading time {0}".format(time() - t0))
    ####################################################################################################################

    regularization = lambd != 0
    if multilabel:
        model = LTLS2_ML.new_model(iterations=it, stop_early=es, validate=validate, regularization=regularization,
                                   lambd=lambd, policy=policy, multiplier=multplier)
    else:
        model = LTLS.new_model(iterations=it, stop_early=es, validate=validate, regularization=regularization,
                               lambd=lambd, policy=policy, multiplier=multplier)
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

    t0 = time()

    Yvalid_hat = model.predict(X_valid)

    logging.info("Prediction time {0}".format(time() - t0))
    valid_prec = precision_score(y_valid, Yvalid_hat, average='micro')
    logging.info("Valid precision@1 {0}".format(valid_prec))

    t0 = time()

    Ytest_hat = model.predict(X_test)

    logging.info("Prediction time {0}".format(time() - t0))
    test_prec = precision_score(y_test, Ytest_hat, average='micro')
    logging.info("Test precision@1 {0}".format(test_prec))

    ####################################################################################################################
    logging.info("Saving model to {0}".format(model_dir))
    model.dump(model_dir)
