
from __future__ import absolute_import, division, print_function

import tflearn
import numpy as np

# Regression data
X=np.random.rand(10,10).tolist()
Y=np.random.rand(10,2).tolist()
print(X)

# Linear Regression graph
input_ = tflearn.input_data(shape=[None,10])
linear = tflearn.fully_connected(input_,256)
linear = tflearn.fully_connected(linear,2)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

X_test=np.random.rand(1,10).tolist()
print(X_test)
print(m.predict(X_test))
