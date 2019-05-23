import multiprocessing
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio
import math
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
tf.reset_default_graph()

# this is the size of our encoded representations
# encoding_dim =16
input_dim=160
def autoencoder(encoding_dim, hidden_units):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(80, activation='relu')(input_img)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dense(hidden_units, activation='relu')(encoded)
    # encoded = LeakyReLU(alpha=0.1)(encoded)
    # encoded = Dense(30, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='linear')(encoded)

    # decoded = Dense(encoding_dim, activation='relu')(encoded)
    # decoded = Dense(30, activation='relu')(encoded)
    decoded = Dense(hidden_units,activation='relu')(encoded)
    # decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(80, activation='relu')(decoded)
    # decoded= LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    #load the data generated in matlab

    # training and validation data between -1 and 1 (30000 training and 10000 validation)

    mat = sio.loadmat('data/Htraining.mat')
    x_train = mat['Htr1']  # array
    mat = sio.loadmat('data/HVALI.mat')
    x_val = mat['H1']  # array
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    # mat = sio.loadmat('data//Htest.mat')
    # x_test = mat['Htest1']  # array
    # x_test = x_test.astype('float32')

    # training and validation data with max gain 0 (50000 training and 20000 validation)

    # mat = sio.loadmat('C://data//Htrlinear.mat')
    # x_train = mat['Htrlinear']  # array
    # mat = sio.loadmat('C://data//Hvalinear.mat')
    # x_val = mat['Hvalinear']  # array
    # x_train = x_train.astype('float32')
    # x_val = x_val.astype('float32')


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
    path = 'resultdeep/TensorBoard_%s' % file

    history=autoencoder.fit(x_train, x_train,
                    epochs=2,
                    batch_size=200,
                    shuffle=True,
                    validation_data=(x_val, x_val),
                    callbacks=[history1,
                               TensorBoard(log_dir=path)])
    history_dict = history.history

    # save the values in excel files
    filename = 'resultdeep/trainloss_{}_{}.csv'.format(file,hidden_units)
    loss_history = np.array(history1.losses_train)
    np.savetxt(filename, loss_history, delimiter=",")

    filename = 'resultdeep/valloss_{}_{}.csv'.format(file,hidden_units)
    loss_history1 = np.array(history1.losses_val)
    np.savetxt(filename, loss_history1, delimiter=",")

    # Plotting the training and validation loss
    loss_values=history_dict['loss']
    val_loss_values=history_dict['val_loss']
    # epochs=range(1,len(loss_values)+1)
    # plt.plot(epochs, loss_values,'bo',label='Training loss')
    # plt.plot(epochs,val_loss_values,'r',label ='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    import multiprocessing

    encoding_dim =[32, 25, 16]
    hidden_units=[60, 50]

    all_args = list()
    for dim in encoding_dim:
        for units in hidden_units:
            all_args.append([dim,units])

    print('number of settings: {}'.format(len(all_args)))

    # on the servers, 12 is a good number for the number of parallel processes
    number_of_parallel_processes = 12
    pool = multiprocessing.Pool(number_of_parallel_processes)
    all_results = pool.starmap(autoencoder, all_args)

    print('The results are:')
    for result in all_results:
         print(result)
    # print('done')
