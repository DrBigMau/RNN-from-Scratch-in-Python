# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 03:02:45 2024

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt

X_t = np.arange(-10,10,0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1)

plt.plot(X_t, Y_t)
plt.show()


from RNN import *

n_neurons = 500
rnn       = RNN(X_t, n_neurons, Tanh())
optimizer = Optimizer_SGD(learning_rate = 1e-5, momentum = 0.9)
T         = rnn.T
n_epoch   = 200
Monitor   = np.zeros((n_epoch,1))



for n in range(n_epoch):

    rnn.forward()

    Y_hat = rnn.Y_hat
    dY    = Y_hat - Y_t
    L     = 0.5*np.dot(dY.T,dY)/T
    
    Monitor[n] = L
    
    rnn.backward(dY)
    
    optimizer.pre_update_params()
    optimizer.update_params(rnn)
    optimizer.post_update_params()
    
    r = n/50
    if r - np.ceil(r) == 0:
        plt.plot(X_t, Y_t)
        plt.plot(X_t, Y_hat)
        plt.legend(['y', '$\hat{y}$'])
        plt.title('epoch ' + str(n))
        plt.show()
        
        
        
plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.title('epoch ' + str(n))
plt.show()

plt.plot(range(n_epoch), Monitor/T)
plt.xlabel('epochs')
plt.ylabel('MSSE')
plt.yscale('log')
plt.show()