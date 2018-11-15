#!/usr/bin/env python3
"""Package for creating test data."""

import argparse
from enum import Enum
import numpy as np
import sys
import os
import csv
import random
import seaborn as sns
import matplotlib.pyplot as plt

SIZE = 190000000

class Distribution(Enum):
    RANDOM = 0
    BINOMIAL = 1
    POISSON = 2
    EXPONENTIAL = 3
    NORMAL = 4
    LOGNORMAL = 5

    @staticmethod
    def from_str(label):
        if label in ('random'):
            return Distribution.RANDOM
        elif label in ('exponential'):
            return Distribution.EXPONENTIAL
        elif label in ('normal'):
            return Distribution.NORMAL
        elif label in ('lognormal'):
            return Distribution.LOGNORMAL
        else:
            raise NotImplementedError

dataPath = "../data/"

# store path
filePath = {
    Distribution.RANDOM: dataPath + "random",
    Distribution.BINOMIAL: dataPath + "binomial",
    Distribution.POISSON: dataPath + "poisson",
    Distribution.EXPONENTIAL: dataPath + "exponential",
    Distribution.NORMAL: dataPath + "normal",
    Distribution.LOGNORMAL: dataPath + "lognormal"
}


def create_data(distribution, data_size=SIZE):
    INT_MAX = np.iinfo(np.uint32).max
    scale = INT_MAX
    retry = 0

    def random_sample(size):
        if distribution == Distribution.RANDOM:
            data = np.arange(data_size)
            np.random.shuffle(data)
        elif distribution == Distribution.BINOMIAL:
            data = np.random.binomial(100, 0.5, size=size)/data_size
        elif distribution == Distribution.POISSON:
            data = np.random.poisson(6, size=size)
        elif distribution == Distribution.EXPONENTIAL:
            data = np.random.exponential(0.15, size=size)
        elif distribution == Distribution.LOGNORMAL:
            data = np.random.lognormal(0, 2, size)/100
        else:
            data = np.random.normal(0.5, 0.1, size=size)

        if distribution != Distribution.RANDOM:
            data = data*scale

        return data.astype(np.uint32)

    data = random_sample(data_size)
    print('first pass size={}'.format(data_size))
    data = np.unique(data)
    print('Unique {}'.format(data.size))

    while data.size < data_size:
        retry += 1
        data = np.append(data, random_sample((data_size-data.size)))
        data = np.unique(data)
        print("Retry: {}".format(retry))
    data = data[:data_size]

    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    with open(filePath[distribution], 'w') as f:
        data.tofile(f)


def load_data(distribution):
    with open(filePath[distribution], 'r') as f:
        data = np.fromfile(f, dtype=np.uint32)
    return data


def graph_data(data):
    sns.distplot(data)


def main(argv):
    # Parse the arguments
    parser = argparse.ArgumentParser(description="""Generate data for testing""")
    parser.add_argument('-s', '--size', help='Size of data to generate', type=int, default=SIZE)
    parser.add_argument('-g', '--graph', help='Graph generated data?', action='store_true')
    args = parser.parse_args(argv[1:])

    dist = [
        Distribution.RANDOM,
        Distribution.EXPONENTIAL,
        Distribution.NORMAL,
        Distribution.LOGNORMAL,
    ]

    for d in dist:
        print(d)
        create_data(d, data_size=args.size)

    if args.graph:
        for d in dist:
            data = load_data(d)
            graph_data(data)
            plt.show()


if __name__ == '__main__':
    main(sys.argv)
