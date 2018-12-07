"""Utilities for the models."""
import time
import numpy as np
import bisect


def clamp(val, minval, maxval):
    """Clamp a value in-between the min and max value."""
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


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


def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1],True])

    # Split input array with those start, stop ones
    out = [a_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return out


def groupby_perID(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1],True])

    # Create cut indices for all unique IDs in b
    n = b_sorted[-1]+2
    cut_idxe = np.full(n, cut_idx[-1], dtype=int)

    insert_idx = b_sorted[cut_idx[:-1]]
    cut_idxe[insert_idx] = cut_idx[:-1]
    cut_idxe = np.minimum.accumulate(cut_idxe[::-1])[::-1]

    # Split input array with those start, stop ones
    out = [a_sorted[i:j] for i,j in zip(cut_idxe[:-1],cut_idxe[1:])]
    return out


if __name__ == '__main__':
    data = [1, 6, 19, 40, 59, 69, 81, 91, 103]
    key = 59
    guess = 4

    tic = time.time()
    # result = linear_search(data, key, guess)
    result = binary_search(data, key, guess, error = 1)
    toc = time.time()

    print('{} = {} at {} time: {}'.format(key, data[result], result, tic-toc))

    data = np.array([1, 6, 19, 40, 59, 69, 81, 91, 103])
    np.random.shuffle(data)
    idx = np.array([3, 3, 1, 3, 1, 2, 2, 4, 3])

    print(groupby(data, idx))
    print(groupby_perID(data, idx))

    split_keys = groupby_perID(data, idx)
    input_key = data

    c_keys = np.concatenate(split_keys)
    a1 = np.argsort(np.argsort(input_key))
    a2 = np.argsort(c_keys)
    r_keys = c_keys[a2][a1]

    print(c_keys[a2])
    print(input_key[a1])

    print('Concat  {}'.format(c_keys))
    print('a1      {}'.format(a1))
    print('a2      {}'.format(a2))
    print('test    {}'.format(r_keys))
    print('correct {}'.format(input_key))



