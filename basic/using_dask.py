from __future__ import division,print_function,absolute_import
import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import *
from tflearn.data_utils import *

# Load Cifar10 DataSet
from tflearn.datasets import cifar10
(X,Y),(X_test,Y_test)=cifar10.load_data("/dataset")
Y=to_categorical(Y,10)
Y_test=to_categorical(Y_test,10)
print(X.shape)
print(Y.shape)
print(X_test.shape)
print(Y_test.shape)


# # Create DASK array using numpy arrays
# import dask.array as da
# X = da.from_array(np.asarray(X), chunks=(1000))
# Y = da.from_array(np.asarray(Y), chunks=(1000))
#
# X_test = da.from_array(np.asarray(X_test), chunks=(1000))
# Y_test = da.from_array(np.asarray(Y_test), chunks=(1000))



# # Create a hdf5 dataset from CIFAR-10 numpy array
import h5py
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('cifar10_X', data=X)
h5f.create_dataset('cifar10_Y', data=Y)
h5f.create_dataset('cifar10_X_test', data=X_test)
h5f.create_dataset('cifar10_Y_test', data=Y_test)
h5f.close()

# Load hdf5 dataset
h5f = h5py.File('data.h5', 'r')
X = h5f['cifar10_X']
Y = h5f['cifar10_Y']
X_test = h5f['cifar10_X_test']
Y_test = h5f['cifar10_Y_test']


# Build network
network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.75)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0,max_checkpoints=30)
model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')
model.save('./cifar10_cnn')
h5f.close()