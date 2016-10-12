# A reprodicton of code for experiments done during the internship.

class Params:
    def __init__(self, path_train, path_validation, path_test, policy, model_dir, it, l, multilabel, es, validate,
                 n_features, multiplier=1):
        self.policy = policy
        self.path_train = path_train
        self.path_validation = path_validation
        self.path_test = path_test
        self.model_dir = model_dir
        self.it = it
        self.l = l
        self.multilabel = multilabel
        self.es = es
        self.validate = validate
        self.n_features = n_features
        self.multiplier = multiplier
