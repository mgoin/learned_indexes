import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, Lambda, add
from keras.models import Model
from keras import backend as K
import numpy as np
import time
import os
import tempfile
import models.utils as utils
import models.train as trainer

def weighted_binary_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    diff = K.maximum(diff-0.499,0)+0.499
    # weight = K.arange(1, 32+1, step=1, dtype=tf.float32)
    # diff = diff * (weight/32)
    return K.mean(diff)

class Learned_AllBits:
    def __init__(self,
                 network_structure=[{'activation': 'relu', 'hidden': 500},]*6,
                 optimizer='adam', loss='mean_squared_error', initializer='glorot_uniform',
                 training_method='start_from_scratch', search_method='linear',
                 batch_size=10000, epochs=100, lr_decay=False, early_stopping=True,
                 **kwargs):
        self.network_structure = network_structure
        self.optimizer = optimizer
        self.loss = loss
        self.initializer = initializer
        self.training_method = training_method
        self.search_method = search_method
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.early_stopping = early_stopping
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
        k = np.asarray(k)
        v = np.asarray(v)
        self.keys = np.append(self.keys, k)
        self.values = np.append(self.values, v)

        # Train model
        if self.training_method == 'train_only_new':
            self.model_train, history = self.train(self.model_train, k, v)
        else:
            self.model_train, history = self.train(self.model_train, self.keys, self.values)
        return history

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

    def train(self, model, keys, values):
        if self.training_method == 'start_from_scratch':
            model.load_weights(self.initial_weights.name)

        bit_values = self.model_int2bit.predict(values).reshape(-1, 32)

        model, train_history = trainer.train_network(model=model, keys=keys, values=bit_values, normalize=False,
                                                     batch_size=self.batch_size, epochs=self.epochs,
                                                     lr_decay=self.lr_decay, early_stopping=self.early_stopping)
        self.train_results = train_history.history
        self._min_error = -1.0
        self._max_error = -1.0
        self._mean_error = -1.0
        return model, train_history.history

    def predict(self, key, batch_size=1000):
        # Key is just a value, make it an array
        if type(key) != np.ndarray:
            key = np.asarray(key).reshape(-1)

        # Get estimate position from the model
        predicted_value = self.model_predict.predict(key, batch_size)
        return predicted_value.flatten()

    def build_network(self):
        input_layer = Input(shape=(1,), dtype=tf.int32)

        # Convert 1 integer input into 32 binary neurons (bits)
        int2bit = Lambda(lambda x: tf.to_float(tf.mod(tf.bitwise.right_shift(tf.expand_dims(x,1), tf.range(32)), 2)), output_shape=(32,))(input_layer)

        x = int2bit
        for layer in self.network_structure:
            x = Dense(layer['hidden'], activation=layer['activation'], kernel_initializer=self.initializer)(x)

        output_layer = Dense(32, activation='relu', kernel_initializer=self.initializer)(x)

        # Convert 32 binary neurons (bits) into 1 integer input
        bit2int = Lambda(lambda x: tf.reduce_sum(tf.bitwise.left_shift(tf.to_int32(tf.rint(tf.clip_by_value(x, 0, 1))), tf.range(32)), 1), output_shape=(1,))(output_layer)

        self.model_int2bit = Model(input_layer, int2bit)
        self.model_predict = Model(input_layer, bit2int)

        self.model_train = Model(input_layer, output_layer)
        self.model_train.compile(optimizer=self.optimizer, loss=self.loss)
        self.model_train.save_weights(self.initial_weights.name)

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
        predicted_values = self.predict(self.keys)
        # Calculate error between predictions and ground truth
        errors = np.abs(predicted_values-self.values)
        self._max_error = np.max(errors)
        self._min_error = np.min(errors)
        self._mean_error = np.mean(errors)

    @property
    def results(self):
        return {
            'type': 'learned_model_allbits',
            'network_structure': self.network_structure,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'training_method': self.training_method,
            'search_method': self.search_method,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr_decay': self.lr_decay,
            'early_stopping': self.early_stopping,
        }

    def save(self, filename='trained_learned_model_fc.h5'):
        self.model_predict.save_weights(filename)

    def load(self, filename='trained_learned_model_fc.h5'):
        self.model_predict.load_weights(filename)
