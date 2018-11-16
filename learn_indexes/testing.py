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
from models import Learned_Model, BTree

class Testing_Framework():
    def __init__(self, model, distribution, sample_size):
        self.model = model
        self.test_distribution = distribution
        self.sample_size = sample_size
        self.train_time = []
        self.inference_time = []

        self.load_test_data()

    def load_test_data(self):
        self.data = load_data(self.test_distribution, self.sample_size)

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

    def time_inference(self, samples=100000):
        tic = time.time()

        val = np.random.choice(self.data, samples)
        found = []
        for v in val:
            x = self.model.predict(v)
            self.model.get(v, x)
        toc = time.time()
        self.inference_time.append((toc-tic)/samples)

def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Run tests""")
    parser.add_argument('-m', '--model', help='name of model to test', default='btree')
    parser.add_argument('-d', '--distribution', help='name of distribution to test', default='random')
    parser.add_argument('-s', '--sample_size', help='number of samples to load for each data', default=10000000)
    args = parser.parse_args(argv[1:])

    # Check the parameters
    if args.model is None:
        print('Model must be specifed', file=sys.stderr)
        exit()
    elif args.model in ('btree'):
        print("Testing with BTree model")
        model = BTree()
    elif args.model in ('learned_fc'):
        print("Testing with Learned FC model")
        model = Learned_Model(network_type='fc')
    elif args.model in ('learned_res'):
        print("Testing with Learned Res model")
        model = Learned_Model(network_type='res')
    else:
        print('Model {} is not recognized.'.format(args.model), file=sys.stderr)
        exit()

    testing_framework = Testing_Framework(model=model, distribution=Distribution.from_str(args.distribution), sample_size=int(args.sample_size))
    testing_framework.run_tests()
    print('Training {}'.format(testing_framework.train_time))
    print('Inference {}'.format(testing_framework.inference_time))

if __name__ == '__main__':
    main(sys.argv)
