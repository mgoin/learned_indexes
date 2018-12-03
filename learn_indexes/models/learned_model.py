from keras.layers import Input, Dense, add
from keras.models import Model
import numpy as np
import time
import models.utils as utils


class Learned_Model:
    def __init__(self,
                 network_type='fc',
                 hidden_activation='relu',
                 hidden_layers=[500, 500, 500, 500, 500],
                 training_method='full', search_method='linear',
                 batch_size=100000, epochs=5):
        self.network_type = network_type
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        self.training_method = training_method
        self.search_method = search_method
        self.batch_size = batch_size
        self.epochs = epochs

        # Set up key/value arrays and build model
        self.clear()

    # Remove all items and reset model
    def clear(self):
        self.keys = np.empty(0, dtype=int)
        self.values = np.empty(0, dtype=int)
        self.min_error = 0.0
        self.max_error = 0.0

        if self.network_type == 'fc':
            self.build_FC()
        elif self.network_type == 'res':
            self.build_Res()
        else:
            raise Exception('Network type "{}" not valid!'.format(self.network_type))

    # Add an item. Return 1 if the item was added, or 0 otherwise.
    def insert(self, key, value):
        # Only accept key if is is greater than all others (append)
        # if np.any(self.keys >= key):
        #     return 0
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
            pos = utils.binary_search(self.keys, key, guess, self.get_max_error())
        else:
            raise Exception('Search method "{}" is not valid!'.format(self.search_method))

        return self.values[pos]

    # Return true if the model contains the given key.
    def has_key(self, key):
        return np.any(self.keys == key)

    # Returns get(key)
    def __getitem__(self, key):
        return self.get(key)

    # Returns insert(key, value)
    def __setitem__(self, key, value):
        return self.insert(key, value)

    # Returns remove(key)
    def __delitem__(self, key):
        return self.remove(key)

    def train(self, model):
        if self.training_method == 'full':
            # load weights from initial build to clear network
            model.load_weights('temp_learned_init.h5')

        x_train = self.keys / float(np.max(self.keys))
        y_train = self.values / float(np.max(self.values))

        # train the network
        model.fit(x_train, y_train,
                  epochs=self.epochs, batch_size=self.batch_size,
                  shuffle=True, verbose=1)
        # save weights of trained network
        model.save_weights('trained_learned.h5')

        # # calculate min and max error over the training set
        # predicted_positions = self.predict(self.keys)
        # y_predicted = np.empty(self.keys.size)
        # for i, p in enumerate(predicted_positions):
        #     y_predicted[i] = self.get(self.keys[i], p)
        # errors = np.abs(y_predicted-y_train)
        # self.max_error = np.max(errors)
        # self.min_error = np.min(errors)

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

    def build_FC(self):
        input_layer = Input(shape=(1,))

        x = input_layer
        for num_neurons in self.hidden_layers:
            x = Dense(num_neurons, activation=self.hidden_activation)(x)

        output_layer = Dense(1, activation='relu')(x)

        self.model = Model(input_layer, output_layer)
        # Compile model and save initial weights for retraining
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.save_weights('temp_learned_init.h5')

    def build_Res(self):
        input_layer = Input(shape=(1,))

        x = input_layer
        for i, num_neurons in enumerate(self.hidden_layers):
            if i%2 == 1:
                if i > 1:
                    x = add([shortcut, x])
                shortcut = x
            x = Dense(num_neurons, activation=self.hidden_activation)(x)

        output_layer = Dense(1, activation='relu')(x)

        self.model = Model(input_layer, output_layer)
        # Compile model and save initial weights for retraining
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.save_weights('temp_learned_init.h5')

    def get_max_error(self):
        return self.max_error

    def get_min_error(self):
        return self.min_error

    @property
    def results(self):
        return {
            'type': 'learned_model_{}'.format(self.network_type),
            'network_type': self.network_type,
            'hidden_activation': self.hidden_activation,
            'hidden_layers': self.hidden_layers,
            'training_method': self.training_method,
            'search_method': self.search_method,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }
