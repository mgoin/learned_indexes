"""Wrapper for the BTree implementation so that it will fit within the testing framework."""
from BTrees.IIBTree import IIBTree
import models.utils as utils

class BTree(IIBTree):
    def __init__(self):
        IIBTree.__init__(self)

    def predict(self, key):
        super().get(self, key)

    def get(self, key, guess):
        return guess


