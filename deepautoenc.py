from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio
import math
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()

# this is the size of our encoded representations
encoding_dim =500
input_dim=4000


input_img = Input(shape=(input_dim,))
encoded = Dense(2000, activation='relu')(input_img)
encoded = Dense(1000, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(1000, activation='relu')(encoded)
decoded = Dense(2000, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

#load the data generated in matlab
mat = sio.loadmat('C://data//HTR.mat')
x_train = mat['HTR1']  # array
mat = sio.loadmat('C://data//HVAL.mat')
x_val = mat['HVAL1']  # array
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
mat = sio.loadmat('C://data//Htest.mat')
x_test = mat['Htest1']  # array
x_test = x_test.astype('float32')

# callback function to save the loss values
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))


history = LossHistory()
file = '_dim' + str(encoding_dim) + time.strftime('_%m_%d')
path = 'resultdeep/TensorBoard_%s' % file

autoencoder.fit(x_train, x_train,
                epochs=2,
                batch_size=200,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[history,
                           TensorBoard(log_dir=path)])

# save the values in excel files
filename = 'resultdeep/trainloss_%s.csv'%file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'resultdeep/valloss_%s.csv'%file
loss_history1 = np.array(history.losses_val)
np.savetxt(filename, loss_history1, delimiter=",")

