# A reprodicton of code for experiments done during the internship.

import logging


def setup_logger(log):
    logging.basicConfig(format='%(message)s', filename=log, level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
