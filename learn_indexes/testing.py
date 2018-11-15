#!/usr/bin/env python3
"""Package for running tests."""

import argparse
from enum import Enum
import numpy as np
import sys
import csv
import random
import time
import matplotlib.pyplot as plt
from create_data import Distribution, load_data
from BTrees.IIBTree import IIBTree

sys.path.append('../')
from models import Learned_FC

class Testing_Framework():
    def __init__(self, model, distribution):
        self.model = model
        self.test_distribution = distribution
        self.train_time = []
        self.inference_time = []

        self.load_test_data()

    def load_test_data(self):
        self.data = load_data(self.test_distribution)

    def run_tests(self, num_tests=1):
        for i in range(num_tests):
            self.time_train()
            self.time_inference()

    def time_train(self):
        def items():
            for i, v in enumerate(self.data):
                yield (int(v), i)

        self.model.clear()

        tic = time.time()
        self.model.update(list(items()))
        toc = time.time()
        self.train_time.append(toc-tic)

    def time_inference(self, samples = 100000):
        tic = time.time()

        val = np.random.choice(self.data, samples)
        found = []
        for v in val:
           self.model.get(v)
        toc = time.time()
        self.inference_time.append((toc-tic)/100)

def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Run tests""")
    parser.add_argument('-m', '--model', help='name of model to test', default='btree')
    parser.add_argument('-d', '--distribution', help='name of distribution to test', default='random')
    args = parser.parse_args(argv[1:])

    # Check the parameters
    if args.model is None:
        print('Model must be specifed', file=sys.stderr)
        exit()
    elif args.model in ('btree'):
        model = IIBTree()
    elif args.model in ('learned_fc'):
        model = Learned_FC()
    else:
        print('Model {} is not recognized.'.format(args.model), file=sys.stderr)
        exit()

    testing_framework = Testing_Framework(model=model, distribution=Distribution.from_str(args.distribution))
    testing_framework.run_tests()
    print('Training {}'.format(testing_framework.train_time))
    print('Inference {}'.format(testing_framework.inference_time))

if __name__ == '__main__':
    main(sys.argv)
