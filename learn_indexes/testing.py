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
from models import Learned_Model, BTree, Hybrid

class Testing_Framework():
    def __init__(self, model, distribution, sample_size, train_percent, inference_samples):
        self.model = model
        self.test_distribution = distribution
        self.sample_size = sample_size
        self.train_time = []
        self.pre_insert_inference_time = []
        self.insert_time = []
        self.post_insert_inference_time = []
        self.train_percent = train_percent
        self.inference_samples = inference_samples

        self.load_test_data()

    def load_test_data(self):
        self.data = load_data(self.test_distribution, self.sample_size)

    @property
    def train_data(self):
        return [(int(v), int(i)) for i, v in enumerate(self.data[:self.split_idx])]

    @property
    def insert_data(self):
        return [(int(v), int(i+self.split_idx)) for i, v in enumerate(self.data[self.split_idx:])]

    @property
    def split_idx(self):
        return int(len(self.data)*self.train_percent)

    def run_tests(self, num_tests=1):
        for i in range(num_tests):
            self.time_train()
            self.time_inference(train_only=True)

            if self.train_percent != 1.0:
                self.time_insert()
                self.time_inference()

    def time_train(self):
        self.model.clear()
        tic = time.time()
        self.model.update(self.train_data)
        toc = time.time()
        self.train_time.append(toc-tic)

    def time_inference(self, train_only=False):

        if train_only:
            k, _ = zip(*self.train_data)
            keys = np.random.choice(k, self.inference_samples)
        else:
            keys = np.random.choice(self.data, self.inference_samples)

        found = []
        tic = time.time()
        guesses = self.model.predict(keys)
        toc = time.time()
        guess_time = (toc-tic)/self.inference_samples

        tic = time.time()
        for k, x in zip(keys, guesses):
            self.model.get(k, x)
        toc = time.time()
        get_time = (toc-tic)/self.inference_samples

        if train_only:
            self.pre_insert_inference_time.append((guess_time, get_time))
        else:
            self.post_insert_inference_time.append((guess_time, get_time))

    def time_insert(self):
        tic = time.time()
        self.model.update(self.insert_data)
        toc = time.time()
        self.insert_time.append(toc-tic)

def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Run tests""")
    parser.add_argument('-m', '--model', help='name of model to test', default='btree')
    parser.add_argument('-d', '--distribution', help='name of distribution to test', default='random')
    parser.add_argument('-s', '--sample-size', help='number of samples to load for each data', default=10000000)
    parser.add_argument('-t', '--train-percent', help='percent of data for initial training (the rest is used for insert)', default=1.0, type=float)
    parser.add_argument('-n', '--number-of-tests', help='number of tests to run', default=1, type=int)
    parser.add_argument('-i', '--inference-samples', help='Number of inferences used for timing', default=100000, type=int)
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
    elif args.model in ('hybrid'):
        print("Testing with Hybrid model")
        model = Hybrid()
    else:
        print('Model {} is not recognized.'.format(args.model), file=sys.stderr)
        exit()

    testing_framework = Testing_Framework(model=model,
                                          distribution=Distribution.from_str(args.distribution),
                                          sample_size=int(args.sample_size),
                                          train_percent=args.train_percent,
                                          inference_samples=args.inference_samples,)

    testing_framework.run_tests(args.number_of_tests)

    print('Split Idx {}'.format(testing_framework.split_idx))
    print('Training {}'.format(testing_framework.train_time))
    print('Pre-Insert Inference {}'.format(testing_framework.pre_insert_inference_time))
    print('Insert {}'.format(testing_framework.insert_time))
    print('Post-Insert Inference {}'.format(testing_framework.post_insert_inference_time))

if __name__ == '__main__':
    main(sys.argv)
