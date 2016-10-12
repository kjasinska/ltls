# A reprodicton of code for experiments done during the internship.

import argparse

import experiment
import predict
from utils.commons import Params
from utils.libsvm_format import sort_libsvm_file
from utils.log import setup_logger


def run():
    parser = argparse.ArgumentParser(description='Run LTLS experiments')
    parser.add_argument('dataset', type=str, help='dataset name')
    args = parser.parse_args()
    path_trai = 'data/{0}/{0}.train'.format(args.dataset)
    path_vali = 'data/{0}/{0}.heldout'.format(args.dataset)
    path_test = 'data/{0}/{0}.test'.format(args.dataset)
    path_trsr = 'data/{0}/{0}.train_sorted'.format(args.dataset)
    path_vasr = 'data/{0}/{0}.heldout_sorted'.format(args.dataset)
    path_tesr = 'data/{0}/{0}.test_sorted'.format(args.dataset)
    model_dir = 'models/{0}'.format(args.dataset)
    log = 'logs/{0}'.format(args.dataset)
    setup_logger(log)

    params = {
        #                       path_trai, path_vali, path_test, policy,    model_dir, it,  l,      multilabel,     es,     validate,   n_features,     multiplier
        'sector': Params(       path_trai, path_vali, path_test, 'ranking', model_dir, 7,   0,      False,          False, True, 55197,     1),
        'aloi.bin': Params(     path_trai, path_vali, path_test, 'ranking', model_dir, 20,  0,      False,          False, True, 636949,    1),
        'LSHTC1': Params(       path_trai, path_vali, path_test, 'ranking', model_dir, 16,  2.3,    False,          False, True, 1199856,   1),
        'imageNet': Params(     path_trai, path_vali, path_test, 'ranking', model_dir, 1,   0,      False,          False, True, 1000,      1),
        'Dmoz': Params(         path_trai, path_vali, path_test, 'ranking', model_dir, 9,   1.75,   False,          False, True, 833484,    1),

        'bibtex': Params(       path_trsr, path_vasr, path_tesr, 'ranking', model_dir, 4,   0,      True,           False, True, 1837,      1),
        'rcv1_regions': Params( path_trsr, path_vasr, path_tesr, 'ranking', model_dir, 7,   0,      True,           False, True, 47237,     3),
        'Eur-Lex': Params(      path_trsr, path_vasr, path_tesr, 'ranking', model_dir, 7,   0,      True,           False, True, 5000,      3),
        'LSHTCwiki': Params(    path_trsr, path_vasr, path_tesr, 'ranking', model_dir, 9,   0,      True,           False, True, 2085167,   4)
    }

    experiment._train(params[args.dataset].path_train, params[args.dataset].path_validation,
                      params[args.dataset].path_test,
                      params[args.dataset].policy, params[args.dataset].model_dir, params[args.dataset].it,
                      params[args.dataset].l, params[args.dataset].multilabel, params[args.dataset].es,
                      params[args.dataset].validate, params[args.dataset].n_features, params[args.dataset].multiplier)

    predict._predict(params[args.dataset].path_test, params[args.dataset].model_dir, params[args.dataset].multilabel,
                     params[args.dataset].n_features)


def sort():
    parser = argparse.ArgumentParser(description='Run LTLS experiments')
    parser.add_argument('dataset', type=str, help='dataset name')
    args = parser.parse_args()
    path_train = 'data/{0}/{0}.train'.format(args.dataset)
    path_validation = 'data/{0}/{0}.heldout'.format(args.dataset)
    path_test = 'data/{0}/{0}.test'.format(args.dataset)
    path_train_sorted = 'data/{0}/{0}.train_sorted'.format(args.dataset)
    path_validation_sorted = 'data/{0}/{0}.heldout_sorted'.format(args.dataset)
    path_test_sorted = 'data/{0}/{0}.test_sorted'.format(args.dataset)

    sort_libsvm_file(path_train, path_train_sorted)
    sort_libsvm_file(path_test, path_test_sorted)
    sort_libsvm_file(path_validation, path_validation_sorted)


if __name__ == "__main__":
    sort()
    run()
