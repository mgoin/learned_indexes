from keras.models import Model
import numpy as np
import time
import models.utils as utils

class Trainer():
    def __init__(self):
        pass

    def train(self, model):
        pass
        # if self.training_method == 'full':
        #     # load weights from initial build to clear network
        #     model.load_weights('temp_learned_init.h5')
        # elif self.training_method == 'partial':
        #     pass
        # else:
        #     raise Exception('"{}" is not a valid training method.'.format(self.training_method))

        # x_train = self.keys / flowat(np.max(self.keys))
        # y_train = self.values / float(np.max(self.values))

        # # train the network
        # model.fit(x_train, y_train,
        #           epochs=self.epochs, batch_size=self.batch_size,
        #           shuffle=True, verbose=1)

        # return model
