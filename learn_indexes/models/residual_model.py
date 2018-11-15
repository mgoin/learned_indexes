from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import bisect

class Learned_Res:
    def __init__(self,
                 search_radius=100,
                 hidden_activation='relu',
                 hidden_layers=[100, 100, 100, 100],
                 training_method='full', batch_size=1000, epochs=10):
        self.search_radius = 100
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        self.training_method = training_method
        self.batch_size = batch_size
        self.epochs = epochs

        # Set up key/value arrays and build model
        self.clear()

    # Remove all items and reset model
    def clear(self):
        self.keys = np.empty(0, dtype=int)
        self.values = np.empty(0, dtype=int)
        self.build_model()

    # Add an item. Return 1 if the item was added, or 0 otherwise.
    def insert(self, key, value):
        # Only accept key if is is greater than all others (append)
        # if np.any(self.keys[:, 0] >= key):
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
        # Get estimate position from the model
        pos = guess

        # Search locally for the key
        k = self.keys[pos]
        left_bound = pos-self.search_radius
        right_bound = pos+self.search_radius
        while k != key:
            i = np.where(self.keys[left_bound:right_bound] == key)[0]
            if i.size == 0:
                left_bound -= self.search_radius
                right_bound += self.search_radius
            else:
                k = self.keys[i[0]]
                pos = i[0]
        return self.values[pos]

    # Return true if the model contains the given key.
    def has_key(key):
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

    def build_model(self):
        input_layer = Input(shape=(1,))

        x = input_layer
        for num_neurons in self.hidden_layers:
            x = Dense(num_neurons, activation=self.hidden_activation)(x)

        output_layer = Dense(1, activation='relu')(x)

        self.model = Model(input_layer, output_layer)
        # Compile model and save initial weights for retraining
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.save_weights('temp_learned_res_init.h5')

    def train(self, model):
        model.load_weights('temp_learned_res_init.h5')
        model.fit(self.keys, self.values,
                  epochs=self.epochs, batch_size=self.batch_size,
                  shuffle=True, verbose=1)
        return model

    def predict(self, x, batch_size=1000):
        return self.model.predict(x, batch_size)
