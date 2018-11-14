"""Package for creating test data."""

from enum import Enum
import numpy as np
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

# store path
filePath = {
    Distribution.RANDOM: "../data/random",
    Distribution.BINOMIAL: "../data/binomial",
    Distribution.POISSON: "../data/poisson",
    Distribution.EXPONENTIAL: "../data/exponential",
    Distribution.NORMAL: "../data/normal",
    Distribution.LOGNORMAL: "../data/lognormal"
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

        # if size > 1:
        #     print(np.max(data))
        #     sns.distplot(data)
        #     plt.show()

        if distribution != Distribution.RANDOM:
            data = data*scale

        # if size > 1:
        #     sns.distplot(data)
        #     plt.show()

        return data.astype(np.uint32)

    data = random_sample(data_size)
    print('first pass {}'.format(data_size))
    data = np.unique(data)
    print('Unique {}'.format(data.size))

    while data.size < data_size:
        retry += 1
        data = np.append(data, random_sample((data_size-data.size)*2))
        data = np.unique(data)
        print("Retrys: {} {}".format(retry, data.size))
    data = data[:data_size]

    with open(filePath[distribution], 'w') as f:
        data.tofile(f)

    print("Retrys: {}".format(retry))

def load_data(distribution):
    with open(filePath[distribution], 'r') as f:
        data = np.fromfile(f, dtype=np.uint32)
    return data


def graph_data(data):
    sns.distplot(data)


if __name__ == '__main__':
    dist = [
        Distribution.RANDOM,
        Distribution.EXPONENTIAL,
        Distribution.NORMAL,
        Distribution.LOGNORMAL,
    ]

    for d in dist:
        print(d)
        # create_data(d, data_size=10000)
        create_data(d)

    # for d in dist:
    #     data = load_data(d)
    #     graph_data(data)
    #     plt.show()

    # with open('../data/lognormal.sorted.190M', 'r') as f:
    #     data = np.fromfile(f, dtype=np.uint32)
    # graph_data(data)
    # plt.show()
