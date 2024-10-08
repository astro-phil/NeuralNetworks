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
    vector[0] = network.net_id
    vector[1] = network.entries
    vector[2] = network.num_layers
    for idx,size in enumerate(network.sizes):
        vector[idx+3] = size
    for weight,biase in zip(network.weights,network.biases):
        vector.extend(weight.flatten())
        vector.extend(biase.flatten())
    return np.array(vector)
    
def vec2net(vector):
    # This function converts the network vector back into the weights and biases matrix
    offset = 0
    new_weights = []
    new_biases = []
    entries = int(vector[1])
    num_layers = int(vector[2])
    sizes = np.array(vector[3:3+num_layers],dtype=np.int8)
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
    return np.array(new_weights),np.array(new_biases),sizes

class Evolution(object):
    def __init__(self,nn_size = (10,6,5),population_size = 2000,num_parents = 100,mutation_rate = 0.1, mutation_strenght = 1,heritage_rate = 0.5,layer_rate = 0.01,node_rate=0.1,insert_factor = 0.1):
        # use this function if you want to train the network via genetics
        # population is the size of the population, num_parents is the number of how many individuals will survise
        self.sizes = nn_size
        self.num_parents = min(population_size-1,max(1,num_parents))
        self.layer_rate = layer_rate
        self.node_rate = node_rate
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
        self.population = np.array(self.population)
        self.target = 0
        self.evolution = 0
        self.population_size = population_size
        self.dna_complexity = len(self.population[0])
        self.insert_factor = insert_factor
        
    def insert_layer(self,layer_idx,vector):
        num_layers = int(vector[2])
        sizes = np.array(vector[3:3+num_layers],dtype=np.int8)
        weights, biases = self.to_net(vector)
        new_sizes = np.insert(sizes,layer_idx+1,sizes[layer_idx+1]+1)
        new_weights = np.eye(sizes[layer_idx+1],sizes[layer_idx+1]+1)
        new_bias = np.zeros(sizes[layer_idx+1])
        new_node = np.zeros(sizes[layer_idx])
        weights[layer_idx] = np.vstack([weights[layer_idx],new_node])
        biases[layer_idx] = np.append(biases[layer_idx],0)
        weights.insert(layer_idx+1,new_weights)
        biases.insert(layer_idx+1,new_bias)
        return self.to_dna(weights, biases, new_sizes)
    
        
    def insert_node(self,layer_idx,vector):
        num_layers = int(vector[2])
        sizes = np.array(vector[3:3+num_layers],dtype=np.int8)
        weights, biases = self.to_net(vector)
        weights[layer_idx-1] = np.vstack([weights[layer_idx-1],np.zeros(sizes[layer_idx-1])])
        weights[layer_idx] = np.hstack([weights[layer_idx],np.zeros(sizes[layer_idx+1]).reshape(-1,1)])
        biases[layer_idx] = np.append(biases[layer_idx],0)
        sizes[layer_idx] += 1
        return self.to_dna(weights, biases, sizes)
        
    def to_net(self,vector):
        # This function converts the network vector back into the weights and biases matrix
        new_weights = []
        new_biases = []
        entries = int(vector[1])
        num_layers = int(vector[2])
        sizes = np.array(vector[3:3+num_layers],dtype=np.int8)
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
        return new_weights,new_biases
        
            
    def to_dna(self,weights,biases,sizes):
        # This function converts all weights and biases into a vector 
        # This vector is needed for the genetic training method
        num_layers = len(weights)+1
        entries = 0
        for idx in range(num_layers-1):
            entries += sizes[idx]*sizes[idx+1]+sizes[idx+1]
        net_id = sum([(idx+1)*size for idx,size in enumerate(sizes)])
        vector = [0] * 20
        vector[0] = net_id
        vector[1] = entries
        vector[2] = num_layers
        for idx,size in enumerate(sizes):
            vector[idx+3] = size
            
        for weight,biase in zip(weights,biases):
            vector.extend(weight.flatten())
            vector.extend(biase.flatten())
        return np.array(vector)
    

    def envolve(self):
        # this is the genetic training method
        # it contains all necessary calculation such as selection ,mating and mutation
        parents = np.empty((self.num_parents,self.dna_complexity))
        fitness = self.fitness.copy()
        generation_fitness = 0
        # Selection of the fittest as followed
        for parent in range(self.num_parents):
            fittest = np.where(fitness==np.max(fitness))
            fittest = fittest[0][0]
            generation_fitness += self.fitness[fittest]
            parents[parent] = self.population[fittest,:]
            fitness[fittest] = -99999999
            
        parents = np.sort(parents,axis=0)
        layer_id = -1
        layer_net_id = -1
        node_id = -1
        node_net_id = -1
        if random.random() < self.layer_rate:
            target_id = random.randint(0,self.num_parents-1)
            layer_net_id = parents[target_id][0]
            num_layers = int(parents[target_id][2])
            sizes = np.array(parents[target_id][3:3+num_layers],dtype=np.int8)
            layer_id = random.randint(0,num_layers-2)
            sizes = np.insert(sizes,layer_id+1,sizes[layer_id+1]+1)
            num_layers += 1
            entries = 0
            for idx in range(num_layers-1):
                entries += sizes[idx]*sizes[idx+1]+sizes[idx+1]
            if entries+20 > self.dna_complexity:
                self.dna_complexity = int(entries+20)
        elif random.random() < self.node_rate:
            target_id = random.randint(0,self.num_parents-1)
            num_layers = int(parents[target_id][2])
            if num_layers > 2:
                node_net_id = parents[target_id][0]
                sizes = np.array(parents[target_id][3:3+num_layers],dtype=np.int8)
                node_id = random.randint(1,num_layers-2)
                sizes[node_id] += 1
                entries = 0
                for idx in range(num_layers-1):
                    entries += sizes[idx]*sizes[idx+1]+sizes[idx+1]
                if entries+20 > self.dna_complexity:
                    self.dna_complexity = int(entries+20)
                
        self.population = np.zeros((self.population_size,self.dna_complexity))
        insert = self.num_parents*self.insert_factor
        for parent in range(self.num_parents):
            if parents[parent][0] == layer_net_id and insert > 0 :
                temp = self.insert_layer(layer_id, parents[parent])
                self.population[parent,:] = np.pad(temp,(0,self.dna_complexity-temp.shape[0]))
                insert -= 1
            elif parents[parent][0] == node_net_id and insert > 0:
                temp = self.insert_node(node_id, parents[parent])
                self.population[parent,:] = np.pad(temp,(0,self.dna_complexity-temp.shape[0]))
                insert -= 1
            else:
                temp = parents[parent] 
                self.population[parent,:] = np.pad(temp,(0,self.dna_complexity-temp.shape[0]))
        
        # Mating the DNA of the fittest to fill up the population as followed
        heritage =  np.array(random.sample(range(20,self.dna_complexity),int(self.heritage_rate*self.dna_complexity)))
        species_count = len(np.unique(parents[:,0]))
        for k in range(self.population_size-self.num_parents):
            idx1 = k%self.num_parents
            idx2 = (k+1)%self.num_parents
            if parents[idx1,0] != parents[idx2,0]:
                idx2 = idx1
            self.population[self.num_parents+k,:] = self.population[idx1,:]
            self.population[self.num_parents+k,heritage] = self.population[idx2,heritage]
        # Mutatating the DNA of the new Generation as followed
        mutation_count = np.uint32((self.mutation_rate*self.dna_complexity))
        mutidx = np.array(random.sample(range(20,self.dna_complexity),mutation_count))
        for idx in range(self.population_size-self.num_parents):
            random_val = np.random.uniform(-1.0,1.0,len(mutidx))
            self.population[self.num_parents+idx,mutidx] = self.population[self.num_parents+idx,mutidx] + random_val*self.mutation_strenght
        self.evolution += 1
        print('Generation number: ' + str(self.evolution) + '  Performance: ' + str(generation_fitness/self.num_parents) + ' Number of Species: ' + str(species_count))
        net_id = parents[0][0]
        count = 0
        for idx,spec in enumerate(parents):
            if spec[0] == net_id:
                count += 1
            else:
                num_layers = int(parents[idx-1,2])
                spec_sizes = parents[idx-1,3:3+num_layers]
                print('--> ' + str(count) + ' Species of ' + str(spec_sizes))
                count = 0
                net_id = spec[0]
        else:
            num_layers = int(spec[2])
            spec_sizes = spec[3:3+num_layers]
            print('--> ' + str(count) + ' Species of ' + str(spec_sizes))
        
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
        self.entries = entries
        self.net_id = sum([(idx+1)*size for idx,size in enumerate(sizes)])
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
        self.entries = entries
        self.weights = weights
        self.biases = biases
        self.net_id = sum([(idx+1)*size for idx,size in enumerate(sizes)])
    
    
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
    
    
net = Sequential((4,3,2))
evo = Evolution()
for x in range(100):
    evo.envolve()