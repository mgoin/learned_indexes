from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import time
import models.utils as utils

def train_network(model, keys, values, normalize,
          training_method='start_from_scratch',
          batch_size=10000, epochs=100,
          lr_decay=False, early_stopping=True):

    if normalize:
        x_train = keys / float(np.max(keys))
        y_train = values / float(np.max(values))
    else:
        x_train = keys
        y_train = values

    # Set up callback functions
    callbacks=[]
    if lr_decay:
        callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.95, patience=10, min_lr=0.00001, verbose=2))
    if early_stopping:
        callbacks.append(EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=2))

    # Train the network
    train_history = model.fit(x_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    shuffle=True, verbose=2, callbacks=callbacks)

    return model, train_history
