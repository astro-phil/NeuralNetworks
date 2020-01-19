# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:02:53 2019

@author: phil-
"""

import random
import pickle
import numpy as np

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
        self.population = []
        
    def init_genetics(self,population = 500,num_parents = 50,mutation_rate = 0.1, mutation_strenght = 1,heritage_rate = 0.5):
        # use this function if you want to train the network via genetics
        # population is the size of the population, num_parents is the number of how many individuals will survise
        self.num_parents = min(population-1,max(1,num_parents))
        self.mutation_rate = mutation_rate
        self.mutation_strenght = mutation_strenght
        self.heritage_rate = heritage_rate
        self.fitness = np.zeros(population)
        for temp in range(population):
            weights = [np.random.randn(y,x) for x , y in zip(self.sizes[:-1],self.sizes[1:])]
            biases = [np.random.randn(y,1) for y in self.sizes[1:]]
            self.population.append(self.net2vec(weights,biases))
        self.population = np.array(self.population)
        self.target = 0
        self.evolution = 0
        
    def init_network(self,weights,biases):
        # use this if you want to initialise the network manually
        self.weights = weights
        self.biases = biases
    
    def softmax(self,z):
        # activation function
        e_x = np.exp(z-np.max(z))
        return e_x/e_x.sum()
    
    def relu(self,z):
        # activation function
        return np.clip(z,0,9999999)
    
    def softplus(self,z):
        # activation function
        return np.log(1+np.exp(z))
    
    def sigmoid(self,z):
        # activation function
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self,z):
        # derivative of the sigmoid function
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def predict(self,activation,function = None):
        # forward propagtoion / prediction of a value by a given initial vector
        # use the function value if you want to change the activation function
        if function == None:
            function = self.sigmoid
        for weight, bias in zip(self.weights,self.biases):
            activation = function(np.dot(weight,activation) + bias)
        return activation
    
    def cost(self,hyp,y):
        # a classic cost function
        return np.sum(np.square(np.subtract(hyp,y)))/len(y)*1/2
    
    def cost_derivative(self,hyp,y):
        # the derivative of the cost function
        return np.subtract(hyp,y)
    
    def backpropagation(self,x,y):
        # backpropagation calculates the changes for the weights and biases and returns them
        # as input it need the input vector as well as the output vector
        activations = [x.reshape(-1,1)]
        neuron_sum = []
        activation = x.reshape(-1,1)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]      
        # forward propagation to get the difference between target and actual vector
        for weight, bias in zip(self.weights,self.biases):
            z = np.dot(weight,activation) + bias
            activation = self.sigmoid(z)
            activations.append(activation)
            neuron_sum.append(z)
        # feeds the diffrence backwards through the network
        delta = self.cost_derivative(activations[-1],y) * self.sigmoid_prime(activations[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for layer_id in range(2,len(self.sizes)):
            z = neuron_sum[-layer_id]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-layer_id+1].transpose(),delta) * sp
            nabla_b[-layer_id] = delta
            nabla_w[-layer_id] = np.dot(delta,activations[-layer_id-1].transpose())
        return nabla_b, nabla_w
    
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
       
    def update_network(self,minibatch,eta):
        # this fucntion updates the weights and biases of the network with backpropagation
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        m = len(minibatch)
        for x,y in minibatch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x,y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b,delta_nabla_b)]
        self.weights = [w-eta/m*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-eta/m*nb for b,nb in zip(self.biases,nabla_b)]
        
    def train(self,training_data,minibatch_size = 10,epochs = 100,eta = 0.1,test_data = None):
        # The Stochastic Gradient Descent training method
        if test_data: 
           n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
           np.random.shuffle(training_data)
           minibatches = [training_data[k:k+minibatch_size] for k in range(0,n,minibatch_size)]
           #generating the minibatches
           for minibatch in minibatches:
               self.update_network(minibatch,eta)
           if test_data:
               print('Epoche {0}: {1} / {2}'.format(j,self.evaluate(test_data),n_test))
           else:
               print('Epoche {0}: completed.'.format(j))
               
    def evaluate(self, test_data):
        # returns the accurency of the network
        test_results = [(np.argmax(self.predict(x.reshape(-1,1))),np.argmax(y)) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
        
    def net2vec(self,weights,biases):
        # This function converts all weights and biases into a vector 
        # This vector is needed for the genetic training method
        vector = []
        for weight,biase in zip(weights,biases):
            vector.extend(weight.flatten())
            vector.extend(biase.flatten())
        return np.array(vector)
        
    def vec2net(self,vector,sizes):
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
        data['weights'] = self.weights
        data['biases'] = self.biases
        data['sizes'] = self.sizes
        data['population'] = self.population
        with open(name+".mdl",'wb+') as handle:
            pickle.dump(data,handle,protocol = pickle.HIGHEST_PROTOCOL)
            handle.close()
            
    def load_model(self,name):
        # loads the network from a file
        with open(name+".mdl",'rb') as handle:
        	data = pickle.load(handle)
        	handle.close()
        self.weights = data['weights']
        self.biases = data['biases']
        self.sizes = data['sizes']
        self.population = data['population'] 
