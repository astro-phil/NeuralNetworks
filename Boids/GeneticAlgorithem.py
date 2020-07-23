# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:22:39 2020

@author: phil-
"""


import numpy as np
from Genetics import *
import pickle
from multiprocessing import Process,freeze_support

GENERATIONS = 10 # number of generation
NUMBER_OF_CORES = 1 # number of processor 
CONTINUE_EVOLUTION = False

INIT_NETWORK_SHAPE = (5,2)
POPULATION_SIZE = 1000
NUMBER_OF_PARENTS = 100
MUTATION_TARGET = 0.25
MUTATION_STREGHT = 1
HERITAGE_TARGET = 0.5
LAYER_CHANCE = 0.1
NODE_CHANCE = 0.2
LAYER_TARGET = 0.5
NODE_TARGET = 0.1



def target_run(targetids,targets,cache):
    ratings = []
    for targetid, target in zip(targetids,targets):
        network = vec2net(target)
        
        ### Use the NN like this
        y = network.predict(x)
        
        ########################################################################
        #                                                                      #
        # Place your Simulation/Calculation or whatever you want to train here #
        #                                                                      #       
        ########################################################################
        ratings.append([targetid," << Insert the Rating of the Network here  >> "])
    with open(cache+".cache",'wb+') as handle:
        pickle.dump(ratings,handle,protocol = pickle.HIGHEST_PROTOCOL)
        handle.close

if __name__ == '__main__':
    freeze_support()
    evolution = Evolution(INIT_NETWORK_SHAPE,POPULATION_SIZE,NUMBER_OF_PARENTS,MUTATION_TARGET,MUTATION_STREGHT,HERITAGE_TARGET,LAYER_CHANCE,NODE_CHANCE,LAYER_TARGET,NODE_TARGET)
    
    if CONTINUE_EVOLUTION:
        evolution.load_model('evolution')
    
    os.makedirs("cache", exist_ok = True) 
    os.makedirs("replay", exist_ok = True) 
    
    for generation in range(GENERATIONS):
        print('Running Year: '+str(generation+1))
        procRunning = []
        for x in range(number_of_cores):
            print("Starting Core: " + str(x))
            targets = range(x,evolution.get_population_size(),number_of_cores)
            proc = Process(target = target_run , args = (np.array(targets),evolution.get_target(targets),"cache/proc_"+str(x)))
            procRunning.append(proc)
            proc.start()
        for proc in procRunning:
            proc.join()
            print("A Process finished.")
        ratings = [] 
        for x in range(number_of_cores):
            with open("cache/proc_"+str(x)+".cache",'rb') as handle:
                ratings.extend(pickle.load(handle))
                handle.close()
        for result in ratings:
            evolution.reward_target(result[1],result[0])
        fittest = evolution.get_fittest_network()
        replaySpecies = evolution.get_target([fittest])
        with open("replay/species_{:03d}.mdl".format(evolution.evolution),'wb+') as handle:
            pickle.dump(replayBoid,handle,protocol = pickle.HIGHEST_PROTOCOL)
            handle.close()
        evolution.save_model('evolution')