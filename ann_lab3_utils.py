import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange
import random

class Hopfield_Network:
    def __init__(self, weights=None):
        """ Hopfield Network
        Args:
            weights: initialize weight matrix manually
        """
        self.weights = weights

    def _sign(self, x):
        """ Sign function: transforms values values to 1 or -1 depending a certain threshold (here: 0)
        Args:
            x: value that is going to be transformed
        """
        x = x.copy()
        update = 1 if x >= 0 else -1
        return update
        
    def fit(self, patterns):
        """ Create weight matrix by (1/N) * X * X.T - I
        Args:
            patterns: memory pattern (x1,x2,...) that are going to be trained
        Notes:
            Weights are calculated by calculating weight matrix for each pattern and then adding them
        """
        weights_pattern = []
        number_units = patterns.shape[1] # number of units
        for i in range(patterns.shape[0]): # for each patterns create weight matrix
            weights = (1/number_units) * ( np.outer(patterns[i].T,patterns[i]) - np.identity(number_units) )
            #print(weights)
            weights_pattern.append(weights)
        weight_matrix = np.sum(weights_pattern,0) # sum all weight matrices of all patterns
        self.weights = weight_matrix
        
    def recall(self, pattern, num_updates=1): # synchronous (batch)
        """ Recall synchronous (batch)
        """
        pattern = pattern.copy()
        recall = pattern.T
        for i in range(num_updates):
            recall = np.dot(self.weights,recall)
            recall[recall >= 0] = 1
            recall[recall < 0] = -1
            #print(recall.reshape(1,pattern.shape[1]))
        return recall.reshape(1,pattern.shape[1])
    
    def recall_sequential_determined(self, pattern, num_updates=1200, ordering="shuffled"): # asynchronous (sequential)
        """ Recall asynchronous (sequential): indices are determined beforehand
        """
        pattern = pattern.copy()
        recall = pattern.T # transform it from (1,n) to (n,1)
        indices = []
        energy_progress = []
        
        if ordering == "ascending":
            indices = [i for i in range(pattern.shape[1])]
        elif ordering == "shuffled":
            indices = [i for i in range(pattern.shape[1])]
            np.random.shuffle(indices)
        else:
            print("Not supported ordering type!")
        
        counter_iter = 0
        counter_engergy = 0 # counter that every 100 iterations the energy gets tracked
        
        for step in range(num_updates):
            
            temp = np.dot(self.weights[indices[counter_iter]].reshape(1,pattern.shape[1]),recall) # returns Matrix shape (1,1)
            update = self._sign(temp)
            recall[indices[counter_iter]][0] = update

            if counter_engergy % 100 == 0: # save the energy after every 100 iterations
                energy_iter = self.energy_function(recall.T)
                energy_progress.append(energy_iter)

            counter_iter+=1
            counter_engergy+=1
            if counter_iter == (len(indices)-1): # if counter reaches last index of array indices, then reset to 0
                counter_iter = 0

        return recall.reshape(1,pattern.shape[1]), energy_progress
    
    def recall_sequential_random(self, pattern, num_updates=1000): # asynchronous (sequential)
        """ Recall asynchronous (sequential): indices are randomly chosen
        """
        pattern = pattern.copy()
        recall = pattern.T # transform it from (1,n) to (n,1)
      
        for i in range(num_updates):
            rand_idx = randrange(pattern.shape[1]) # random index gets choosen
            temp = np.dot(self.weights[rand_idx].reshape(1,pattern.shape[1]),recall) # Matrix with shape (1,1)
            update = self._sign(temp)#1 if temp[0][0] >= 0 else -1
            recall[rand_idx][0] = update

        return recall.reshape(1,pattern.shape[1])
    
    def energy_function(self,pattern):
        """ Calculate energy of the respective pattern, a.k.a Lyapunov function or energy function
        """
        pattern = pattern.copy()
        energy_value = 0
        
        for i in range(self.weights.shape[0]): # row
            for j in range(self.weights.shape[1]): # column
                energy_value += (-1) * self.weights[i][j] * pattern[0][i] * pattern[0][j]
        
        return energy_value
    
    def create_distortion(self, pattern, flipping_ratio=0.4):
        """ Distort a given pattern  by randomly flipping a selected number of units
        """
        pattern = pattern.copy()
        indices = indices = [i for i in range(pattern.shape[1])]
        np.random.shuffle(indices)
        indices = indices[0:int(flipping_ratio*pattern.shape[1])]
        
        for i in indices:
            pattern[0][i] = -1 if pattern[0][i] == 1 else 1      #random.choice([-1,1])
        
        return pattern
    
    def calculate_distortion_ratio(self,original,distorted):
        """ Calculating the distortion: How many values are different
        """
        counter = 0
        total = original.shape[1]
        for i in range(total):
            if original[0][i] == distorted[0][i]:
                counter+=1
        
        return (1-(counter/total))

