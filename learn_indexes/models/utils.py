"""Utilities for the models."""
import time
import numpy as np
import bisect

def binary_search(data, key, guess, error = 100):
    """Perform binary search to find the keys location in an array.

    The guess and error parameters are used to narrow the search window and speed
    up the search for large arrays."""

    guess = np.clip(guess, 0, len(data)-1)
    value = guess

    minv = np.clip(value - error, 0, len(data)-1)
    maxv = np.clip(value + error, 0, len(data)-1)

    while True:
        value = bisect.bisect_left(data, key, minv, maxv)

        if value == minv and minv != 0:
            minv = value - error
            maxv = value + error

        elif value == maxv and maxv != len(data)-1:
            minv = value - error
            maxv = value + error

        else:
            return value


def linear_search(data, key, guess):
    """Perform linear search to find the keys location in an array.

    The guess is used as the starting location in the search."""

    guess = np.clip(guess, 0, len(data)-1)
    value = guess

    # While the key is not found
    while True:

        # If the key is found, then return the key
        if data[value] == key:
            return value;

        # If the data is greater than the key, decrease the value
        elif data[value] > key:
            # if the value is no longer changing in the right direction, return close value.
            if value > guess or value == 0:
                return value
            else:
                value -= 1

        # If the data is less than the key, increase the value
        else: # data[value] < key:
            # if the value is no longer changing in the right direction, return close value.
            if value < guess or value + 1 == len(data):
                return value
            else:
                value += 1

if __name__ == '__main__':
    data = [1, 6, 19, 40, 59, 69, 81, 91, 103]
    key = 59
    guess = 4

    tic = time.time()
    # result = linear_search(data, key, guess)
    result = binary_search(data, key, guess, error = 1)
    toc = time.time()

    print('{} = {} at {} time: {}'.format(key, data[result], result, tic-toc))


