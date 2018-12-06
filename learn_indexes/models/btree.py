"""Wrapper for the BTree implementation so that it will fit within the testing framework."""
from BTrees.IIBTree import IIBTree
import models.utils as utils
import numpy as np


class BTree(IIBTree):
    def __init__(self):
        IIBTree.__init__(self)

    def predict(self, key):
        # Key is just a value, make it an array
        if type(key) != np.ndarray:
            key = np.full(1, key)

        guess = np.zeros(len(key))
        for i, k in enumerate(key):
            guess[i] = super().get(int(k))
        return guess

    def get(self, key, guess):
        return guess

    def update(self, collection):
        int_list = [(int(i), int(j)) for i, j in collection]
        super().update(int_list)

    @property
    def results(self):
        return {'type': 'btree'}

    @property
    def max_error(self):
        return 0.0

    @property
    def min_error(self):
        return 0.0

    @property
    def mean_error(self):
        return 0.0


if __name__ == '__main__':
    b = BTree()
    u = [(1, 2), (3, 40)]
    b.update(u)
    print(b.predict(1))
    print(b.predict(3))
