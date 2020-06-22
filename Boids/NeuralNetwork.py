# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:02:53 2019

@author: phil-
"""

import random
import pickle
import numpy as np
import os

def net2vec(weights,biases):
    # This function converts all weights and biases into a vector 
    # This vector is needed for the genetic training method
    vector = []
    for weight,biase in zip(weights,biases):
        vector.extend(weight.flatten())
        vector.extend(biase.flatten())
    return np.array(vector)
    
def vec2net(vector,sizes):
    # This function converts the network vector back into the weights and biases matrix
    offset = 0
    new_weights = []
    new_biases = []
    for x,y in zip(sizes[:-1],sizes[1:]):
        w_long = x*y
        w = vector[offset:offset+w_long]
        w = w.reshape(y,x)
        offset += w_long
        b_long = y
        b = vector[offset:offset + b_long]
        b = b.reshape(y,-1)
        offset += b_long
        new_weights.append(w)
        new_biases.append(b)
    return np.array(new_weights),np.array(new_biases)

class Evolution(object):
    def __init__(self,nn_size = (10,5),population = 2000,num_parents = 100,mutation_rate = 0.1, mutation_strenght = 1,heritage_rate = 0.5):
        # use this function if you want to train the network via genetics
        # population is the size of the population, num_parents is the number of how many individuals will survise
        self.sizes = nn_size
        self.num_parents = min(population-1,max(1,num_parents))
        self.mutation_rate = mutation_rate
        self.mutation_strenght = mutation_strenght
        self.heritage_rate = heritage_rate
        self.fitness = np.zeros(population)
        self.population = []
        for temp in range(population):
            weights = [np.random.randn(y,x) for x , y in zip(self.sizes[:-1],self.sizes[1:])]
            biases = [np.random.randn(y,1) for y in self.sizes[1:]]
            self.population.append(net2vec(weights,biases))
        self.population = np.array(self.population)
        self.target = 0
        self.evolution = 0

    def envolve(self):
        # this is the genetic training method
        # it contains all necessary calculation such as selection ,mating and mutation
        parents = np.empty((self.num_parents,self.population.shape[1]))
        fitness = self.fitness.copy()
        generation_fitness = 0
        # Selection of the fittest as followed
        for parent in range(self.num_parents):
            fittest = np.where(fitness==np.max(fitness))
            fittest = fittest[0][0]
            generation_fitness += self.fitness[fittest]
            parents[parent,:] = self.population[fittest,:]
            fitness[fittest] = -99999999
        # Mating the DNA of the fittest to fill up the population as followed
        offspring_size = (self.population.shape[0]-self.num_parents,self.population.shape[1])
        offspring = np.empty(offspring_size)
        hertiage =  np.array(random.sample(range(0,offspring.shape[1]),int(self.heritage_rate*offspring.shape[1])))
        for k in range(offspring_size[0]):
            idx1 = k%self.num_parents
            idx2 = (k+1)%self.num_parents
            offspring[k,:] = parents[idx1,:]
            offspring[k,hertiage] = parents[idx2,hertiage]
        # Mutatating the DNA of the new Generation as followed
        mutation_count = np.uint32((self.mutation_rate*offspring.shape[1]))
        mutidx = np.array(random.sample(range(0,offspring.shape[1]),mutation_count))
        for idx in range(offspring_size[0]):
            random_val = np.random.uniform(-1.0,1.0,len(mutidx))
            offspring[idx,mutidx] = offspring[idx,mutidx] + random_val*self.mutation_strenght
        self.population[0:self.num_parents,:] = parents
        self.population[self.num_parents:,:] = offspring
        self.evolution += 1
        print('Generation number: ' + str(self.evolution) + '  Performance: ' + str(generation_fitness/self.num_parents))

        
    def get_fittest_network(self):
        # this function return the fittest network of the current population
        fittest = np.where(self.fitness==np.max(self.fitness))
        fittest = fittest[0][0]
        return fittest
    
    def get_population_size(self):
        # returns the population size
        return len(self.population)
    
    def reward_target(self,score,target):
        # this method is needed to set score/fintess of the current network
        # its important to use this function on every iteration
        self.fitness[target] = score

    def set_target(self,target):
        # loads a target network from the population into the current network
        if target >= len(self.population) or target < 0:
            return False
        weights,biases = self.vec2net(self.population[target],self.sizes)
        self.weights = weights
        self.biases = biases
        return True
    
    def get_target(self,target):
        # returns the network vector from a given target id
        return self.population[target]
    
    def save_model(self,name):
        # saves the current network to a file (includes also the population)
        data = {}
        data['sizes'] = self.sizes
        data['population'] = self.population
        data['evolution'] = self.evolution
        with open(name+".mdl",'wb+') as handle:
            pickle.dump(data,handle,protocol = pickle.HIGHEST_PROTOCOL)
            handle.close()
            
    def load_model(self,name):
        # loads the network from a file
        if not os.path.isfile(name+".mdl"):
            return
        with open(name+".mdl",'rb') as handle:
        	data = pickle.load(handle)
        	handle.close()
        self.sizes = data['sizes']
        self.population = data['population'] 
        self.evolution = data['evolution']
        
    

class NeuralNetwork(object):
    ''' 
    A classical neural network with stochastic gradient descent and genetics as training methods
    it has various activationfunctions such as
    relu, sigmoid, stable softmax, softplus
    but only sigmoid works with sgd
    '''
    def __init__(self,sizes):
        self.sizes = sizes
        self.weights = np.array([np.random.randn(y,x) for x , y in zip(sizes[:-1],sizes[1:])])
        self.biases = np.array([np.random.randn(y,1) for y in sizes[1:]])
        # Random initialization of the weights and biases for all layers
        
        
    def init_network(self,weights,biases):
        # use this if you want to initialise the network manually
        self.weights = weights
        self.biases = biases
    
    
    def relu(self,z):
        # activation function
        return np.clip(z,0,99999)
    
    
    def predict(self,activation,function = None):
        # forward propagtoion / prediction of a value by a given initial vector
        # use the function value if you want to change the activation function
        if function == None:
            function = self.relu
        for weight, bias in zip(self.weights,self.biases):
            activation = function(np.dot(weight,activation) + bias)
        return activation
    
    def cost(self,hyp,y):
        # a classic cost function
        return np.sum(np.square(np.subtract(hyp,y)))/len(y)*1/2
    
    def cost_derivative(self,hyp,y):
        # the derivative of the cost function
        return np.subtract(hyp,y)
    
    def relu_derivative(self,hyp,y):
        # the derivative of the cost function
        return np.subtract(hyp,hyp*y)
    

