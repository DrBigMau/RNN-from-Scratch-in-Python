# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:37:03 2024

@author: MMH_user
"""

import numpy as np
import matplotlib.pyplot as plt

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1)

plt.plot(X_t, Y_t)
plt.show()

###############################################################################
#X(t) --> Y(t)

from MyRNN import *

rnn =  RunMyRNN(X_t, Y_t, Tanh())

X_new = np.arange(0, 20, 0.3)
X_new = X_new.reshape(len(X_new), 1)

Y_hat = ApplyMyRNN(X_new, rnn)

plt.plot(X_t, Y_t)
plt.plot(X_new, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()
    
###############################################################################
#Y(t) --> Y(t + dt)

from MyRNN import *

dt  = 50
rnn =  RunMyRNN(Y_t, Y_t, Tanh(), n_epoch = 800, n_neurons = 100, decay = 0.1,\
                dt = dt)

Y_hat = ApplyMyRNN(Y_t, rnn)

X_t   = np.arange(len(Y_t))

plt.plot(X_t, Y_t)
plt.plot(X_t + dt, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()







