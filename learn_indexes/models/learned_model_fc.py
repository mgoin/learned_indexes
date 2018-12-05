from keras.layers import Input, Dense, add
from keras.models import Model
import numpy as np
import time
import os
import tempfile
import models.utils as utils

class Learned_FC:
    def __init__(self,
                 network_structure=[{'activation': 'relu', 'hidden': 500},
                                    {'activation': 'relu', 'hidden': 500},
                                    {'activation': 'relu', 'hidden': 500},
                                    {'activation': 'relu', 'hidden': 500},
                                    {'activation': 'relu', 'hidden': 500},],
                 optimizer='adam', loss='mean_squared_error',
                 training_method='start_from_scratch', search_method='linear',
                 batch_size=100000, epochs=5, **kwargs):
        self.network_structure = network_structure
        self.optimizer = optimizer
        self.loss = loss
        self.training_method = training_method
        self.search_method = search_method
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_parameters = kwargs

        self.initial_weights = tempfile.NamedTemporaryFile(delete=False)
        self.initial_weights.close()

        # Set up key/value arrays and build model
        self.clear()

    def __del__(self):
        os.remove(self.initial_weights.name)

    # Remove all items and reset model
    def clear(self):
        self.keys = np.empty(0, dtype=int)
        self.values = np.empty(0, dtype=int)
        self.build_network()
         
        self._min_error = -1.0
        self._max_error = -1.0
        self._mean_error = -1.0

    # Add an item. Return 1 if the item was added, or 0 otherwise.
    def insert(self, key, value):
        self.keys = np.append(self.keys, key)
        self.values = np.append(self.values, value)
        return 1

    # Remove an item. Return 1 if removed successful, or 0 otherwise.
    def remove(self, key):
        index, = np.where(self.keys == key)
        if index.size == 0:
            return 0
        self.keys = np.delete(self.keys, index)
        self.values = np.delete(self.values, index)
        return 1

    # Add the items from the given collection.
    def update(self, collection):
        # Add items to model
        k, v = zip(*collection)
        self.keys = np.append(self.keys, k)
        self.values = np.append(self.values, v)
        # Retrain model
        self.model = self.train(self.model)

    # Return the value or the default if the key is not found.
    def get(self, key, guess):
        # Search locally for the key, starting from the guess
        if self.search_method == 'linear':
            pos = utils.linear_search(self.keys, key, guess)
        elif self.search_method == 'binary':
            pos = utils.binary_search(self.keys, key, guess, self.max_error)
        else:
            raise Exception('Search method "{}" is not valid!'.format(self.search_method))

        return self.values[pos]

    # Return true if the model contains the given key.
    def has_key(self, key):
        return np.any(self.keys == key)

    # Returns get(key)
    def __getitem__(self, key):
        p = self.predict(key)
        return self.get(key, p)

    # Returns insert(key, value)
    def __setitem__(self, key, value):
        return self.insert(key, value)

    # Returns remove(key)
    def __delitem__(self, key):
        return self.remove(key)

    def train(self, model):
        if self.training_method == 'start_from_scratch':
            model.load_weights(self.initial_weights.name)
        elif self.training_method == 'start_from_previous':
            pass
        else:
            raise Exception('"{}" is not a valid training method.'.format(self.training_method))

        x_train = self.keys / float(np.max(self.keys))
        y_train = self.values / float(np.max(self.values))

        # train the network
        model.fit(x_train, y_train,
                  epochs=self.epochs, batch_size=self.batch_size,
                  shuffle=True, verbose=1)

        return model

    def predict(self, key, batch_size=1000):
        # Key is just a value, make it an array
        if type(key) != np.ndarray:
            key = np.full(1, key)

        normalized_key = key / float(np.max(self.keys))

        # Get estimate position from the model
        normalized_pos = self.model.predict(normalized_key, batch_size)

        # Convert the normalized position back to index
        pos = (normalized_pos * float(np.max(self.values))).astype(int)
        return pos.flatten()

    def build_network(self):
        input_layer = Input(shape=(1,))

        x = input_layer
        for layer in self.network_structure:
            x = Dense(layer['hidden'], activation=layer['activation'])(x)

        output_layer = Dense(1, activation='relu')(x)

        self.model = Model(input_layer, output_layer)
        # Compile model and save initial weights for retraining
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.save_weights(self.initial_weights.name)

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
        predicted_positions = self.predict(self.keys)
        y_train = self.values / float(np.max(self.values))
        y_predicted = np.empty(self.values.size)
        for i, p in enumerate(predicted_positions):
            y_predicted[i] = self.get(self.keys[i], p)
        errors = np.abs(y_predicted-y_train)
        self._max_error = np.max(errors)
        self._min_error = np.min(errors)
        self._mean_error = np.mean(errors)

    @property
    def results(self):
        return {
            'type': 'learned_model_fc',
            'network_structure': self.network_structure,
            'training_method': self.training_method,
            'search_method': self.search_method,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }

    def save(self, filename='trained_learned_model_fc.h5'):
        self.model.save_weights(filename)

    def load(self, filename='trained_learned_model_fc.h5'):
        self.model.load_weights(filename)