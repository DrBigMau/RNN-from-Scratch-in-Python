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

n_neurons = 500

from RNN import *

rnn   = RNN(X_t, n_neurons,Tanh())
rnn.forward()

Y_hat = rnn.Y_hat
H     = rnn.H
T     = rnn.T

dY         = Y_hat - Y_t
L          = 0.5*np.dot(dY.T,dY)/T

plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()

for h in H:
    plt.plot(np.arange(20), h[0:20], 'k-', linewidth = 1, alpha = 0.05)
