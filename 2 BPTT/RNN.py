# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 02:24:29 2024

@author: MMH_user
"""

import numpy as np

class RNN():
    
    def __init__(self, X_t, n_neurons, Activation):
        
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
        
        self.Activation = Activation
    
    
    def forward(self):
        
        #initializing dweights
        self.dWx       = np.zeros((self.n_neurons, 1))
        self.dWh       = np.zeros((self.n_neurons, self.n_neurons))
        self.dWy       = np.zeros((1, self.n_neurons))
        self.dbiases   = np.zeros((self.n_neurons, 1))
    
        Activation     = self.Activation
        X_t            = self.X_t
        H              = self.H
        Y_hat          = self.Y_hat
        ht             = H[0]# initial state vector
    
        ACT            = [Activation for i in range(self.T)]
        
        [ACT,H,Y_hat]  = self.RNNCell(X_t, ht, ACT, H, Y_hat)
        
        self.Y_hat     = Y_hat
        self.H         = H
        self.ACT       = ACT
    
    def RNNCell(self, X_t, ht, ACT, H, Y_hat):
        
        for t, xt in enumerate(X_t):
            
            xt = xt.reshape(1,1)
                        
            out = np.dot(self.Wx, xt) + np.dot(self.Wh, ht)\
                  + self.biases
                  
            ACT[t].forward(out)
            ht  = ACT[t].output

            y_hat_t  = np.dot(self.Wy, ht)
            
            H[t+1]   = ht
            Y_hat[t] = y_hat_t
            
        return(ACT,H,Y_hat)
    
###############################################################################
#
###############################################################################
class Tanh:
        
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.inputs = inputs
            
    def backward(self, dvalues):
        deriv        = 1 - self.output**2
        self.dinputs = np.multiply(deriv, dvalues)
###############################################################################
#
###############################################################################