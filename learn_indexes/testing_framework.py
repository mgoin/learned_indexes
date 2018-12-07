#!/usr/bin/env python3
"""Package for running tests."""

import argparse
import csv
import os
import random
import sys
import time
import datetime
from enum import Enum
import numpy as np
import tensorflow as tf
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
from BTrees.IIBTree import IIBTree

import utils.datastore as ds
from create_data import Distribution, load_data
from models import Learned_FC, Learned_Res, Learned_Bits, BTree, Hybrid, Hybrid_Original

RESULTS_DIR = '../results'


class Testing_Framework():
    def __init__(self, model, distribution, sample_size, train_percent, inference_samples, seed=None, num_tests=1):
        self.model = model
        self.test_distribution = distribution
        self.sample_size = sample_size
        self.train_time = []
        self.pre_insert_inference_time = []
        self.insert_time = []
        self.post_insert_inference_time = []
        self.train_percent = train_percent
        self.inference_samples = inference_samples
        self.pre_insert_min_error = []
        self.pre_insert_max_error = []
        self.pre_insert_mean_error = []
        self.post_insert_min_error = []
        self.post_insert_max_error = []
        self.post_insert_mean_error = []
        self.training_history = []
        self.seed = seed
        self.num_tests = num_tests

        # Seed before and after since load test data could be using cached results
        np.random.seed(self.seed)
        self.load_test_data()
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

    def load_test_data(self):
        self.data = load_data(self.test_distribution, self.sample_size)

    @property
    def train_data(self):
        return [(v, np.int32(i)) for i, v in enumerate(self.data[:self.split_idx])]

    @property
    def insert_data(self):
        return [(v, np.int32(i+self.split_idx)) for i, v in enumerate(self.data[self.split_idx:])]

    @property
    def split_idx(self):
        return int(len(self.data)*self.train_percent)

    @property
    def results(self):
        results = {
            'model': self.model.results,
            'test_distribution': self.test_distribution.name,
            'sample_size': self.sample_size,
            'train_time': self.train_time,
            'pre_insert_inference_time': self.pre_insert_inference_time,
            'insert_time': self.insert_time,
            'post_insert_inference_time': self.post_insert_inference_time,
            'train_percent': self.train_percent,
            'inference_samples': self.inference_samples,
            'collection_time': datetime.datetime.now(),
            'pre_insert_min_error': self.pre_insert_min_error,
            'pre_insert_max_error': self.pre_insert_max_error,
            'pre_insert_mean_error': self.pre_insert_mean_error,
            'post_insert_min_error': self.post_insert_min_error,
            'post_insert_max_error': self.post_insert_max_error,
            'post_insert_mean_error': self.post_insert_mean_error,
            'seed': self.seed,
            'num_tests': self.num_tests,
            'training_history': self.training_history,
        }
        return results

    def save_results(self):
        filename = '{model[type]}_{test_distribution}_{sample_size}'.format(**self.results)
        filename = os.path.join(RESULTS_DIR, filename)
        actual_filename = ds.save_json(filename, self.results, False)

        print("Results saved to {}.".format(actual_filename))

    def run_tests(self):
        for i in range(self.num_tests):
            train_hist = []

            hist = self.time_train()
            train_hist.append(hist)
            self.time_inference(train_only=True)
            self.pre_insert_mean_error.append(self.model.mean_error)
            self.pre_insert_max_error.append(self.model.max_error)
            self.pre_insert_min_error.append(self.model.min_error)

            if self.train_percent != 1.0:
                hist = self.time_insert()
                train_hist.append(hist)
                self.time_inference()
                self.post_insert_mean_error.append(self.model.mean_error)
                self.post_insert_max_error.append(self.model.max_error)
                self.post_insert_min_error.append(self.model.min_error)

            self.training_history.append(train_hist)

            K.clear_session()


    def time_train(self):
        self.model.clear()
        tic = time.time()
        history = self.model.update(self.train_data)
        toc = time.time()
        self.train_time.append(toc-tic)
        return history

    def time_inference(self, train_only=False):

        if train_only:
            k, _ = zip(*self.train_data)
            keys = np.random.choice(k, self.inference_samples)
        else:
            keys = np.random.choice(self.data, self.inference_samples)

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
        history = self.model.update(self.insert_data)
        toc = time.time()
        self.insert_time.append(toc-tic)
        return history


def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Run tests""")
    parser.add_argument('-m', '--model', help='name of model to test', default='btree')
    parser.add_argument('-d', '--distribution', help='name of distribution to test', default='random')
    parser.add_argument('-s', '--sample-size', help='number of samples to load for each data', default=10000000)
    parser.add_argument('-t', '--train-percent', help='percent of data for initial training (the rest is used for insert)', default=1.0, type=float)
    parser.add_argument('-n', '--number-of-tests', help='number of tests to run', default=1, type=int)
    parser.add_argument('-i', '--inference-samples', help='Number of inferences used for timing', default=100000, type=int)
    parser.add_argument('--seed', help='Seed for the random number generator', default=None, type=int)
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
        model = Learned_FC()
    elif args.model in ('learned_res'):
        print("Testing with Learned Res model")
        model = Learned_Res()
    elif args.model in ('learned_bits'):
        print("Testing with Learned Bits model")
        model = Learned_Bits()
    elif args.model in ('hybrid'):
        print("Testing with Hybrid model")
        model = Hybrid()
    elif args.model in ('hybrid_original'):
        print("Testing with the original Hybrid model")
        model = Hybrid_Original()
    else:
        print('Model {} is not recognized.'.format(args.model), file=sys.stderr)
        exit()

    testing_framework = Testing_Framework(model=model,
                                          distribution=Distribution.from_str(args.distribution),
                                          sample_size=int(args.sample_size),
                                          train_percent=args.train_percent,
                                          inference_samples=args.inference_samples,
                                          seed=args.seed)

    testing_framework.run_tests()
    testing_framework.save_results()

    print('Split Idx {}'.format(testing_framework.split_idx))
    print('Training {}'.format(testing_framework.train_time))
    print('Pre-Insert Inference {}'.format(testing_framework.pre_insert_inference_time))
    print('Insert {}'.format(testing_framework.insert_time))
    print('Post-Insert Inference {}'.format(testing_framework.post_insert_inference_time))


if __name__ == '__main__':
    main(sys.argv)
