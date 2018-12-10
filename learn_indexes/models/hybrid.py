import gc

import numpy as np

import models.utils as utils
from .btree import BTree
from .learned_model_fc import Learned_FC


class Hybrid:
    def __init__(self, index=None, stage_nums=[1, 10], search_method='linear', model=Learned_FC, **kwargs):
        self.index = index
        self._keys = np.empty(0, dtype=np.int32)
        self._values = np.empty(0, dtype=np.int32)
        self.search_method = search_method
        self.stage_nums = stage_nums
        self.model_parameters = kwargs
        self.model=model

        self._min_error = -1.0
        self._max_error = -1.0
        self._mean_error = -1.0

    def clear(self):
        self.index = None
        self._keys = np.empty(0, dtype=np.int32)
        self._values = np.empty(0, dtype=np.int32)

        self._min_error = -1.0
        self._max_error = -1.0
        self._mean_error = -1.0

    def insert(self, key, value):
        self._keys = np.append(self._keys, key)
        self._values = np.append(self._values, value)
        self.index, train_results = Hybrid._hybrid_training(
            self.stage_nums,
            train_data_x=self._keys,
            train_data_y=self._values,
            test_data_x=[],
            test_data_y=[],
            model_cls=self.model,
            **self.model_parameters,
        )

        return train_results

    def update(self, collection):
        k, v = zip(*collection)
        self._keys = np.append(self._keys, k)
        self._values = np.append(self._values, v)

        self.index, train_results = Hybrid._hybrid_training(
            self.stage_nums,
            train_data_x=self._keys,
            train_data_y=self._values,
            test_data_x=[],
            test_data_y=[],
            model_cls=self.model,
            **self.model_parameters,
        )

        return train_results

    # Return the value or the default if the key is not found.
    def get(self, key, guess):
        # Search locally for the key, starting from the guess
        if self.search_method == 'linear':
            pos = utils.linear_search(self._keys, key, guess)
        elif self.search_method == 'binary':
            pos = utils.binary_search(self._keys, key, guess, self.max_error)
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
        predictions = [None for _ in range(len(stage_nums) + 1)]
        predictions[0] = np.zeros(len(input_key), dtype=np.int32)

        # For each stage
        for s in range(len(stage_nums)):
            # Split keys by the node to perform the prediction
            split_keys = utils.groupby_perID(input_key, predictions[s])

            # For each node perform the prediction
            split_predictions = [np.empty(0, dtype=np.int32) for _ in range(len(split_keys))]
            for i in range(len(split_keys)):
                if self.index[s][i] is None:
                    continue
                if len(split_keys[i]) == 0:
                    continue
                split_predictions[i] = self.index[s][i].predict(split_keys[i])

                # Do not clip the last stage
                if s != len(stage_nums) - 1:
                    np.clip(split_predictions[i], 0, stage_nums[s + 1] - 1, out=split_predictions[i])

            # Recombine the predictions to the original order by sorting the original and recombined keys
            c_keys = np.concatenate(split_keys)
            a1 = np.argsort(np.argsort(input_key))
            a2 = np.argsort(c_keys)
            c_predictions = np.concatenate(split_predictions)
            predictions[s+1] = c_predictions[a2][a1]

        return predictions[-1]

    @property
    def results(self):
        results = {
            'type': 'hybrid',
            'search_method': self.search_method,
            'stage_numbers': self.stage_nums,
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
        if self._max_error < 0:
            self.calculate_error()
        return self._max_error

    @property
    def min_error(self):
        if self._min_error < 0:
            self.calculate_error()
        return self._min_error

    @property
    def mean_error(self):
        if self._mean_error < 0:
            self.calculate_error()
        return self._mean_error

    def calculate_error(self):
        # Get predicted positions from model
        predicted_values = self.predict(self._keys)
        # Calculate error between predictions and ground truth
        errors = np.abs(predicted_values-self._values)
        self._max_error = np.max(errors)
        self._min_error = np.min(errors)
        self._mean_error = np.mean(errors)

    @staticmethod
    def _hybrid_training(stage_nums, train_data_x, train_data_y, test_data_x, test_data_y, model_cls, threshold=-1, **kwargs):
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
        index_training = [[None for _ in range(stage_nums[i])] for i in range(stage_length)]

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
                index[i][j] = model_cls(**kwargs)
                index_training[i][j] = index[i][j].update(zip(inputs, labels))
                index_training[i][j]['min_error'] = index[i][j].min_error
                index_training[i][j]['mean_error'] = index[i][j].mean_error
                index_training[i][j]['max_error'] = index[i][j].max_error

                # If the stage is not the last stage
                # if i < M then
                if i < stage_length - 1:

                    # Pick model in next stage with output of this model.
                    # The next stage model does not have to match the training label.
                    # p = index[i][j](r.key) / stages[i + 1];
                    p = index[i][j].predict(tmp_inputs[i][j])

                    # Clamp the stage to the valid range
                    # p = utils.clamp(int(round(p)), 0, stage_nums[i + 1] - 1)
                    # np.rint(p, out=p)
                    np.clip(p, 0, stage_nums[i + 1] - 1, out=p)

                    # allocate data into training set for models in next stage
                    # for r ∈ tmp records[i][j] do
                    for ind in range(len(tmp_inputs[i][j])):
                        # Add the input and labels to the next stage
                        # tmp_records[i + 1][p].add(r);
                        tmp_inputs[i + 1][p[ind]].append(tmp_inputs[i][j][ind])
                        tmp_labels[i + 1][p[ind]].append(tmp_labels[i][j][ind])

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
            if threshold > 0 and mean_abs_err > threshold:
                # replace model with BTree if mean error > threshold
                print("Using BTree")
                # index[M][j] = new B-Tree trained on tmp_records[M][j];
                index[stage_length - 1][i] = BTree()
                index_training[stage_length - 1][i] = index[stage_length - 1][i].update(list(zip(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])))

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
            'stage_distribution': stage_distribution,
            'stage_training': index_training,
        }

        # return index;
        return index, results
