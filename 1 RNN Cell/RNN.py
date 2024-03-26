# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 02:24:29 2024

@author: MMH_user
"""

import numpy as np

class RNN():
    
    def __init__(self, X_t, n_neurons):
        
        self.T         = max(X_t.shape)
        self.X_t       = X_t
        #inizializing prediction vector of yt
        self.Y_hat     = np.zeros((self.T, 1))
            
        self.n_neurons = n_neurons
        
        self.Wx        = 0.1*np.random.randn(n_neurons, 1)
        self.Wh        = 0.1*np.random.randn(n_neurons, n_neurons)
        self.Wy        = 0.1*np.random.randn(1, n_neurons)
        self.biases    = 0.1*np.random.randn(n_neurons, 1)
        
        self.H         = [np.zeros((self.n_neurons,1)) for t in range(self.T+1)]
    
    
    def forward(self, xt, ht_1):
        
        out     = np.dot(self.Wx, xt) + np.dot(self.Wh, ht_1) + self.biases
        ht      = np.tanh(out)
        y_hat_t = np.dot(self.Wy, ht)
        
        return ht, y_hat_t, out