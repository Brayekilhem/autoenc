from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.reset_default_graph()

# this is the size of our encoded representations
encoding_dim = 500
input_dim=4000

# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
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


history1 = LossHistory()
file = '_dim' + str(encoding_dim) + time.strftime('_%m_%d')
path = 'result1/TensorBoard_%s' % file

history = autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=200,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=[history1,
                           TensorBoard(log_dir=path)])
history_dict = history.history
# save the values in excel files
filename = 'result1/trainloss_%s.csv'%file
loss_history = np.array(history1.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result1/valloss_%s.csv'%file
loss_history = np.array(history1.losses_val)
np.savetxt(filename, loss_history, delimiter=",")

# x_hat=autoencoder.predict(x_test)
# x_test_real=x_test(:,0:(np.shape(x_test)[1])//2)
# x_test_imag=x_test(:,((np.shape(x_test)[1])//2):np.shape(x_test))
# x_test_c=x_test_real+1j*x_hat_imag
#
# x_hat_real=x_hat(:,0:(np.shape(x_hat)[1])//2)
# x_hat_imag=x_hat(:,((np.shape(x_hat)[1])//2):np.shape(x_hat))
# x_hat_c=x_hat_real+1j*x_hat_imag
#
# pow=np.sum(abs(x_test_c)**2, axis=1)
# pow1=np.sum(abs(x_hat_c)**2, axis=1)
# mse = np.sum(abs(x_test_c-x_hat_c)**2, axis=1)
# print("NMSE is ", 10*math.log10(np.mean(mse/pow)))

# Plotting the training and validation loss
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)
plt.plot(epochs, loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'r',label ='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

