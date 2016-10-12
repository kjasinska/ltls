# A reprodicton of code for experiments done during the internship.

import argparse
import logging
from time import time

from sklearn.metrics import precision_score

from ltls import LTLS, LTLS2_ML
from utils.io import load_dataset
from utils.log import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='LTLS training arguments.')
    parser.add_argument('path', type=str, help='data')
    parser.add_argument('n_features', type=int, help='number of features')
    parser.add_argument('--log', type=str, default=None, help='log file')
    parser.add_argument('--model_dir', type=str, default=None, help='A directory where to store the model')
    parser.add_argument('--multilabel', help='multilabel classification, default: False (multiclass)',
                        action='store_true')
    args = parser.parse_args()
    return args.path, args.log, args.model_dir, args.multilabel, args.n_features


def _predict(path, model_dir, multilabel, n_features):
    t0 = time()
    logging.info("Reading model {0}".format(model_dir))

    if multilabel:
        model = LTLS2_ML.load(model_dir)
    else:
        model = LTLS.load(model_dir)

    logging.info("Reading time {0}".format(time() - t0))
    ####################################################################################################################
    t0 = time()
    logging.info("Reading file {0}".format(path))

    X, Y, _, _, le = load_dataset(path, n_features, multilabel=multilabel, classes_=model.classes_)

    logging.info("Num classes in train+validation {0}".format(le.num_in_training))
    logging.info("Num unseen classes in test {0}".format(le.num_new_in_test))

    logging.info("Reading time {0}".format(time() - t0))
    ####################################################################################################################

    Y_hat = model.predict(X)

    logging.info("Prediction time {0}".format(time() - t0))
    train_prec = precision_score(Y, Y_hat, average='micro')
    logging.info("Precision@1 {0}".format(train_prec))


def predict():
    path, log, model_dir, multilabel, n_features = parse_args()
    setup_logger(log)
    _predict(path, model_dir, multilabel, n_features)


if __name__ == "__main__":
    predict()
