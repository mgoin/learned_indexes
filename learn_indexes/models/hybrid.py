import gc

import numpy as np

import models.utils as utils
from .btree import BTree
from .learned_model_fc import Learned_FC


class Hybrid:
    def __init__(self, index=None, search_method='linear', **kwargs):
        self.index = index
        self._keys = np.empty(0, dtype=int)
        self._values = np.empty(0, dtype=int)
        self.search_method = search_method
        self.train_results = None
        self.model_parameters = kwargs

    def clear(self):
        self.index = None
        self._keys = np.empty(0, dtype=int)
        self._values = np.empty(0, dtype=int)

    def insert(self, key, value):
        self._keys = np.append(self._keys, key)
        self._values = np.append(self._values, value)
        self.index, self.train_results = Hybrid._hybrid_training(
            threshold=[1, 4],
            use_threshold=[True, False],
            stage_nums=[1, 10],
            train_data_x=self._keys,
            train_data_y=self._values,
            test_data_x=[],
            test_data_y=[],
            **self.model_parameters,
        )

    def update(self, collection):
        k, v = zip(*collection)
        self._keys = np.append(self._keys, k)
        self._values = np.append(self._values, v)

        self.index, self.train_results = Hybrid._hybrid_training(
            threshold=[1, 4],
            use_threshold=[True, False],
            stage_nums=[1, 10],
            train_data_x=self._keys,
            train_data_y=self._values,
            test_data_x=[],
            test_data_y=[],
            **self.model_parameters,
        )

    # Return the value or the default if the key is not found.
    def get(self, key, guess):
        # Search locally for the key, starting from the guess
        if self.search_method == 'linear':
            pos = utils.linear_search(self._keys, key, guess)
        elif self.search_method == 'binary':
            pos = utils.binary_search(self._keys, key, guess, self.get_max_error())
        else:
            raise Exception('Search method "{}" is not valid!'.format(self.search_method))

        return self._values[pos]

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def predict(self, input_key):

        # Key is just a value, make it an array
        if type(input_key) != np.ndarray:
            input_key = np.full(1, input_key)

        stage_nums = [len(i) for i in self.index]
        predictions = np.zeros_like(input_key)

        # For each key
        for i, key in enumerate(input_key):
            # For each stage
            p = 0
            for s in range(len(stage_nums) - 1):
                # Pick the model in the next stage
                p = self.index[s][p].predict(key)[0]

                # Clamp the stage to the valid range
                p = utils.clamp(int(round(p)), 0, stage_nums[s + 1] - 1)

            # Predict position
            predictions[i] = self.index[len(stage_nums) - 1][p].predict(key)

        return predictions

    @property
    def results(self):
        results = {
            'type': 'hybrid',
            'search_method': self.search_method,
            'train_results': self.train_results,
        }

        models = [[] for _ in self.index]

        for i, stage in enumerate(self.index):
            for model in stage:
                if model is None:
                    models[i].append(None)
                else:
                    models[i].append(model.results)

        results['models'] = models

        return results

    @property
    def max_error(self):
        return 0.0

    @property
    def min_error(self):
        return 0.0

    @property
    def mean_error(self):
        return 0.0

    @staticmethod
    def _hybrid_training(threshold, use_threshold, stage_nums, train_data_x, train_data_y, test_data_x, test_data_y,
                        **kwargs):
        """Hybrid training structure, 2 stages

        Input: int threshold, int stages[], NN complexity
        Data: record data[], Model index[][]
        Result: trained index

        Notes:
            stage_nums is stages in paper
            stage_length is M in paper
            records is inputs and labels
            The remaining arguments are NN complexity.
        """
        # M = stages.size;
        stage_length = len(stage_nums)

        # Result: trained index
        index = [[None for _ in range(stage_nums[i])] for i in range(stage_length)]

        # tmp_records[][];
        tmp_inputs = [[[] for _ in range(stage_nums[i])] for i in range(stage_length)]
        tmp_labels = [[[] for _ in range(stage_nums[i])] for i in range(stage_length)]
        divisor = [[1.0 for _ in range(stage_nums[i])] for i in range(stage_length)]

        # tmp_records[][] = all data;
        tmp_inputs[0][0] = train_data_x
        tmp_labels[0][0] = train_data_y
        test_inputs = test_data_x

        # Each node in the hybrid tree
        # for i ← 1 to M do
        for i in range(stage_length):
            # for j ← 1 to stages[i] do
            for j in range(stage_nums[i]):
                # skip nodes with no labels
                if len(tmp_labels[i][j]) == 0:
                    print("Skipping node i:{}, j:{}".format(i, j))
                    continue

                # Setup records for training
                inputs = tmp_inputs[i][j]
                labels = []
                test_labels = []

                # first stage, calculate how many models in next stage
                # Labels then hold the correct index into the next stage.
                if i < stage_length - 1:
                    divisor[i][j] = stage_nums[i + 1] * 1.0 / (max(tmp_labels[i][j]) - min(tmp_labels[i][j]))
                    for k in tmp_labels[i][j]:
                        labels.append(utils.clamp(int(k * divisor[i][j]), 0, stage_nums[i + 1] - 1))
                    for k in test_data_y:
                        test_labels.append(utils.clamp(int(k * divisor[i][j]), 0, stage_nums[i + 1] - 1))
                else:
                    labels = tmp_labels[i][j]
                    test_labels = test_data_y

                # index[i][j] = new NN trained on tmp_records[i][j];
                index[i][j] = Learned_FC(**kwargs)
                index[i][j].update(zip(inputs, labels))

                # If the stage is not the last stage
                # if i < M then
                if i < stage_length - 1:
                    # allocate data into training set for models in next stage
                    # for r ∈ tmp records[i][j] do
                    for ind in range(len(tmp_inputs[i][j])):
                        # Pick model in next stage with output of this model.
                        # The next stage model does not have to match the training label.
                        # p = index[i][j](r.key) / stages[i + 1];
                        p = index[i][j].predict(tmp_inputs[i][j][ind])[0]

                        # Clamp the stage to the valid range
                        p = utils.clamp(int(round(p)), 0, stage_nums[i + 1] - 1)

                        # Add the input and labels to the next stage
                        # tmp_records[i + 1][p].add(r);
                        tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                        tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

        # Replace model with BTree if performance is below a threshold.
        # for j ← 1 to index[M].size do
        for i in range(stage_nums[stage_length - 1]):
            # Continue if invalid index
            if index[stage_length - 1][i] is None:
                print("Invalid index i:{}".format(i))
                continue

            # index[M][j].calc err(tmp_records[M][j]);
            mean_abs_err = index[stage_length - 1][i].mean_error

            # if index[M][j].max_abs_err > threshold then
            if mean_abs_err > threshold[stage_length - 1]:
                # replace model with BTree if mean error > threshold
                print("Using BTree")
                # index[M][j] = new B-Tree trained on tmp_records[M][j];
                index[stage_length - 1][i] = BTree()
                index[stage_length - 1][i].update(list(zip(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])))

        # Store model distributions
        stage_distribution = []
        for s in range(stage_length):
            stage = []
            print('stage {}: '.format(s), end='')

            for node in range(stage_nums[s]):
                stage.append(len(tmp_inputs[s][node]))
                print(' {}'.format(len(tmp_inputs[s][node])), end='')
            stage_distribution.append(stage)
            print()

        results = {
            'stage_distribution': stage_distribution
        }

        # return index;
        return index, results
