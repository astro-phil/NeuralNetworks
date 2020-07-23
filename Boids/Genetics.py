# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:02:53 2019

@author: phil-
"""

import random
import pickle
import numpy as np
import os


def net2vec(network):
    # This function converts all weights and biases into a vector 
    # This vector is needed for the genetic training method
    vector = [0] * 20
    vector[0] = len(vector)-20
    vector[1] = network.num_layers
    for idx,size in enumerate(network.sizes):
        vector[idx+2] = size
    for weight,biase in zip(network.weights,network.biases):
        vector.extend(weight.flatten())
        vector.extend(biase.flatten())
    return np.array(vector)
    
def vec2net(vector):
    # This function converts the network vector back into the weights and biases matrix
    offset = 0
    new_weights = []
    new_biases = []
    entries = int(vector[0])
    num_layers = int(vector[1])
    sizes = np.array(vector[2:2+num_layers],dtype=np.int32)
    dna = vector[20:20+entries]
    for x,y in zip(sizes[:-1],sizes[1:]):
        w_long = x*y
        w = dna[offset:offset+w_long]
        w = w.reshape(y,x)
        offset += w_long
        b_long = y
        b = dna[offset:offset + b_long]
        b = b.reshape(y,-1)
        offset += b_long
        new_weights.append(w)
        new_biases.append(b)
    return new_weights , new_biases , sizes

class Evolution(object):
    def __init__(self,nn_size = (10,6,5),population_size = 2000,num_parents = 100,mutation_rate = 0.1, mutation_strenght = 1,heritage_rate = 0.5,layer_rate = 0.01,node_rate=0.1,layer_strength = 0.1, node_strength= 0.4):
        # use this function if you want to train the network via genetics
        # population is the size of the population, num_parents is the number of how many individuals will survise
        self.sizes = nn_size
        self.num_parents = min(population_size-1,max(1,num_parents))
        self.layer_rate = layer_rate
        self.node_rate = node_rate
        self.layer_strength = layer_strength
        self.node_strength = node_strength
        self.mutation_rate = mutation_rate
        self.mutation_strenght = mutation_strenght
        self.heritage_rate = heritage_rate
        self.fitness = np.zeros(population_size)
        self.population = []
        self.species_id = 0
        for temp in range(population_size):
            weights = [np.random.randn(y,x) for x , y in zip(self.sizes[:-1],self.sizes[1:])]
            biases = [np.random.randn(y,1) for y in self.sizes[1:]]
            self.population.append(self.to_dna(weights,biases,nn_size))
        self.population = self.population
        self.target = 0
        self.evolution = 0
        self.population_size = population_size
        self.dna_complexity = len(self.population[0])
        
    def insert_layer(self,layer_idx,vector):
        weights, biases, sizes = self.to_net(vector)
        new_sizes = np.insert(sizes,layer_idx+1,sizes[layer_idx+1]+1)
        new_weights = np.eye(sizes[layer_idx+1],sizes[layer_idx+1]+1)
        new_bias = np.zeros(sizes[layer_idx+1])
        new_node = np.zeros(sizes[layer_idx])
        weights[layer_idx] = np.vstack([weights[layer_idx],new_node])
        biases[layer_idx] = np.vstack([biases[layer_idx],[0]])
        weights.insert(layer_idx+1,new_weights)
        biases.insert(layer_idx+1,new_bias)
        return self.to_dna(weights, biases, new_sizes)
    
        
    def insert_node(self,layer_idx,vector):
        weights, biases, sizes = self.to_net(vector)
        weights[layer_idx-1] = np.vstack([weights[layer_idx-1],np.zeros(sizes[layer_idx-1])])
        weights[layer_idx] = np.hstack([weights[layer_idx],np.zeros(sizes[layer_idx+1]).reshape(-1,1)])
        biases[layer_idx-1] = np.vstack([biases[layer_idx-1],[0]])
        sizes[layer_idx] += 1
        return self.to_dna(weights, biases, sizes)

        
    def to_net(self,vector):
        # This function converts the network vector back into the weights and biases matrix
        new_weights = []
        new_biases = []
        entries = int(vector[0])
        num_layers = int(vector[1])
        sizes = np.array(vector[2:2+num_layers],dtype=np.int32)
        dna = vector[20:20+entries]
        offset = 0
        for x,y in zip(sizes[:-1],sizes[1:]):
            w_long = x*y
            w = dna[offset:offset+w_long]
            w = w.reshape(y,x)
            offset += w_long
            b_long = y
            b = dna[offset:offset + b_long]
            b = b.reshape(y,-1)
            offset += b_long
            new_weights.append(w)
            new_biases.append(b)
        return new_weights,new_biases, sizes
        
            
    def to_dna(self,weights,biases,sizes):
        # This function converts all weights and biases into a vector 
        # This vector is needed for the genetic training method
        num_layers = len(weights)+1
        vector = [0] * 20
        vector[1] = num_layers
        for idx,size in enumerate(sizes):
            vector[idx+2] = size
            
        for weight,biase in zip(weights,biases):
            vector.extend(weight.flatten())
            vector.extend(biase.flatten())
        vector[0] = len(vector)-20
        return np.array(vector)
    

    def envolve(self):
        # this is the genetic training method
        # it contains all necessary calculation such as selection ,mating and mutation
        population = []
        fitness = self.fitness.copy()
        generation_fitness = 0
        # Selection of the fittest as followed
        for parent in range(self.num_parents):
            fittest = np.where(fitness==np.max(fitness))
            fittest = fittest[0][0]
            generation_fitness += self.fitness[fittest]
            population.append(self.population[fittest])
            fitness[fittest] = -99999999
            
        population.sort(key=lambda x: x[0])
        
        # Mating the DNA of the fittest to fill up the population as followed
        for k in range(self.population_size-self.num_parents):
            idx1 = k%self.num_parents
            idx2 = (k+1)%self.num_parents
            if population[idx1][0] != population[idx2][0]:
                idx2 = idx1
            heritage = np.array(random.sample(range(int(population[idx1][0])),int(self.heritage_rate*population[idx1][0])),dtype=np.int32) + 20
            child = np.empty(population[idx1].shape)
            child[:] = population[idx1][:]
            child[heritage] = population[idx2][heritage]
            population.append(child)
        # Mutatating the DNA of the new Generation as followed
        layeridx = np.array(random.sample(range(self.population_size),int(self.population_size*self.layer_strength)),dtype=np.int32)
        layerprop = random.random()
        nodeidx = np.array(random.sample(range(self.population_size),int(self.population_size*self.node_strength)),dtype=np.int32)
        nodeprop = random.random()
        for idx in range(self.population_size-self.num_parents):
            num_layers = population[self.num_parents+idx][1]            
            if idx in layeridx and layerprop < self.layer_rate:
                layerindx = random.randint(0,num_layers-2)
                child = self.insert_layer(layerindx,population[self.num_parents+idx])
                population[self.num_parents+idx] = child
            elif idx in nodeidx and num_layers > 2 and nodeprop < self.node_rate:
                layerindx = random.randint(1,num_layers-2)
                child = self.insert_node(layerindx,population[self.num_parents+idx])
                population[self.num_parents+idx] = child
            
        for idx in range(self.population_size-self.num_parents):
            mutidx = np.array(random.sample(range(int(population[self.num_parents+idx][0])),int(self.heritage_rate*population[self.num_parents+idx][0])),dtype=np.int32)+20
            random_val = np.random.uniform(-self.mutation_strenght,self.mutation_strenght,int(population[self.num_parents+idx][0]))
            population[self.num_parents+idx][mutidx] += random_val[mutidx-20]
        population.sort(key=lambda x: x[0])
        self.population = population
        self.evolution += 1
        print('Generation number: ' + str(self.evolution) + '  Performance: ' + str(generation_fitness/self.num_parents))
        net_id = self.population[0][0]
        count = 0
        for idx in range(len(self.population)):
            if self.population[idx][0] == net_id:
                count += 1
            else:
                num_layers = int(self.population[idx-1][1])
                spec_sizes = self.population[idx-1][2:2+num_layers]
                print('--> ' + str(count) + ' Species of ' +str(spec_sizes) + ' with id ' + str(net_id))
                count = 1
                net_id = self.population[idx][0]
        else:
            num_layers = int(self.population[-1][1])
            spec_sizes = self.population[-1][2:2+num_layers]
            print('--> ' + str(count) + ' Species of ' + str(spec_sizes)+ ' with id ' + str(self.population[-1][0]))
        
    def get_fittest_network(self):
        # this function return the fittest network of the current population
        fittest = np.where(self.fitness==np.max(self.fitness))
        fittest = fittest[0][0]
        return fittest
    
    def get_population_size(self):
        # returns the population size
        return self.population_size
    
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
    
    def get_target(self,targets_id):
        # returns the network vector from a given target id
        targets = []
        for idx in targets_id:
            targets.append(self.population[idx])
        return targets
    
    def save_model(self,name):
        # saves the current network to a file (includes also the population)
        data = {}
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
        self.population = data['population'] 
        self.evolution = data['evolution']
        
    

class Sequential(object):
    ''' 
    A classical neural network with stochastic gradient descent and genetics as training methods
    it has various activationfunctions such as
    relu, sigmoid, stable softmax, softplus
    but only sigmoid works with sgd
    '''
    def __init__(self,sizes=(1,1)):
        self.sizes = sizes
        self.num_layers = len(sizes)
        entries = 0
        for idx in range(self.num_layers-1):
            entries += sizes[idx]*sizes[idx+1]+sizes[idx+1]
        self.net_id = entries
        self.weights = [np.random.randn(y,x) for x , y in zip(sizes[:-1],sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # Random initialization of the weights and biases for all layers
        
        
    def init_network(self,weights,biases,sizes):
        # use this if you want to initialise the network manually
        self.sizes = sizes
        self.num_layers = len(sizes)
        entries = 0
        for idx in range(self.num_layers-1):
            entries += sizes[idx]*sizes[idx+1]+sizes[idx+1]
        self.weights = weights
        self.biases = biases
        self.net_id = entries
    
    
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
    