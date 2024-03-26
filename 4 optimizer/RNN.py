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
        self.Y_hat      = np.zeros((self.T, 1))
            
        self.n_neurons  = n_neurons
        
        self.Wx         = 0.1*np.random.randn(n_neurons, 1)
        self.Wh         = 0.1*np.random.randn(n_neurons, n_neurons)
        self.Wy         = 0.1*np.random.randn(1, n_neurons)
        self.biases     = 0.1*np.random.randn(n_neurons, 1)
        
        self.H          = [np.zeros((self.n_neurons,1)) for t in range(self.T+1)]
        
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
    
    
    def backward(self, dvalues):
        
        #dY = dinputs
        
        T       = self.T
        H       = self.H
        X_t     = self.X_t
        
        ACT     = self.ACT
        
        dWx     = self.dWx
        dWy     = self.dWy
        dWh     = self.dWh
        Wy      = self.Wy
        Wh      = self.Wh
        
        dht     = np.dot(Wy.T, dvalues[-1].reshape(1,1))
        
        dbiases = self.dbiases
        
        #actual BPTT
        for t in reversed(range(T)):
            
            dy = dvalues[t].reshape(1,1)
            xt = X_t[t].reshape(1,1)
            
            ACT[t].backward(dht)
            dtanh = ACT[t].dinputs
            
            dWx     += np.dot(dtanh,xt)
            dWy     += np.dot(H[t+1],dy).T
            dWh     += np.dot(H[t],dtanh.T)
            dbiases += dtanh
            
            dht = np.dot(Wh, dtanh) + np.dot(Wy.T,dy)

        self.dWx     = dWx
        self.dWy     = dWy
        self.dWh     = dWh
        self.dbiases = dbiases
        
        self.H       = H

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
class Optimizer_SGD:
    #initializing with a default learning rate of 0.1
    def __init__(self, learning_rate = 1e-5, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'Wx_momentums'):
                layer.Wx_momentums     = np.zeros_like(layer.Wx)
                layer.Wy_momentums     = np.zeros_like(layer.Wy)
                layer.Wh_momentums     = np.zeros_like(layer.Wh)
                layer.bias_momentums   = np.zeros_like(layer.biases)
                
            #now the momentum parts
            Wx_updates = self.momentum * layer.Wx_momentums - \
                self.current_learning_rate * layer.dWx
            layer.Wx_momentums = Wx_updates
            
            Wy_updates = self.momentum * layer.Wy_momentums - \
                self.current_learning_rate * layer.dWy
            layer.Wy_momentums = Wy_updates
            
            Wh_updates = self.momentum * layer.Wh_momentums - \
                self.current_learning_rate * layer.dWh
            layer.Wh_momentums = Wh_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            
            Wx_updates     = -self.current_learning_rate * layer.dWx
            Wy_updates     = -self.current_learning_rate * layer.dWy
            Wh_updates     = -self.current_learning_rate * layer.dWh
            bias_updates   = -self.current_learning_rate * layer.dbiases
        
        layer.Wx      += Wx_updates 
        layer.Wy      += Wy_updates 
        layer.Wh      += Wh_updates 
        layer.biases  += bias_updates 
        
    def post_update_params(self):
        self.iterations += 1



