# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:06:34 2020

@author: phil-
"""
import numpy as np
from Genetics import *
import pickle
from multiprocessing import Process,freeze_support

class Field(object):
    def __init__(self,size = (1000,1000),limit = 100):
        self.size = size
        self.limit = limit
        self.corners = np.array([(0,0),(0,size[1]),(size[0],size[1]),(size[0],0)])
        self.normals = np.array([(1,0),(0,-1),(-1,0),(0,1)])
    
    def get_direction(self,dirvec,posx,posy):
        direction = 0
        if posx < self.limit:
            direction = np.copysign(1,np.dot(self.normals[1],dirvec)) * (self.limit-posx)/self.limit 
        if posy < self.limit:
            direction = np.copysign(1,np.dot(self.normals[0],dirvec)) * (self.limit-posy)/self.limit
        if posx > self.size[0] - self.limit:
            direction = np.copysign(1,np.dot(self.normals[3],dirvec)) * (posx - self.size[0] + self.limit)/self.limit
        if posy > self.size[1] - self.limit:
            direction = np.copysign(1,np.dot(self.normals[2],dirvec)) * (posy - self.size[1] + self.limit)/self.limit
        return direction * 20
        

    def get_size(self):
        return self.size
        
     
class Boid(object):
    def __init__(self,idx = 0,field = None,others = None,ki = None,init_rot = 0,init_pos = (0,0),init_speed = 1,view = 100):
        self.rot = init_rot 
        self.pos = np.array(init_pos,dtype=np.float16)
        self.speed = init_speed
        self.field = field
        self.others = others
        self.view = view
        self.cache = np.zeros((len(others),5))
        self.count = 0
        self.idx = idx
        self.ki = ki
        self.rating = 0
        self.beacon = 0
        
    def update(self):
        c,s = np.cos(self.rot), np.sin(self.rot)
        rotm = np.matrix([[c, -s], [s, c]])
        irotm = np.matrix([[c, s], [-s, c]])
        dirvec = np.asarray(np.dot(rotm,[1,0])).reshape(-1)
        self.pos[0] += self.speed*c
        self.pos[1] += self.speed*s
        self.get_nearest()
        
        drot = self.field.get_direction(dirvec,self.pos[0],self.pos[1])*0.005 *self.speed
        if drot == 0:
            self.flocking(irotm)
        else:
            self.rot += drot
        
        #self.pos = self.field.get_position(self.pos)
        #self.flocking(irotm)
            
        if self.rot < 0:
            self.rot = 2*np.pi
        elif self.rot >= 2*np.pi:
            self.rot = 0
            
    
    def get_nearest(self):
        self.count = 0
        for boid in self.others:
            if self.idx == boid.idx:
                continue
            if self.count >= self.cache.shape[0]:
                break
            dpos = self.pos-boid.pos
            dist = np.sqrt(dpos[0]**2+dpos[1]**2)
            if dist < self.view:
                self.cache[self.count][0] = boid.rot
                self.cache[self.count][1] = boid.speed
                self.cache[self.count][2] = dist
                self.cache[self.count][3] = boid.pos[0]
                self.cache[self.count][4] = boid.pos[1]
                
                self.count += 1
    
    def flocking(self,irotm):
        if self.count == 0:
            return
        beaboid = np.sum(self.cache[0:self.count],axis=0)/self.count
        min_dist = self.cache[0:self.count,2]
        minboid = self.cache[np.argmin(min_dist)]
        beavector = np.dot(irotm,beaboid[3:]-self.pos).reshape(-1)
        minvector = np.dot(irotm,minboid[3:]-self.pos).reshape(-1)

        
        drot = beaboid[0]-self.rot
        if drot > np.pi:
            drot = -2+drot/np.pi
            
        mindrot = minboid[0]-self.rot
        if mindrot > np.pi:
            mindrot = -2+mindrot/np.pi
            
        beaboid[0] = drot
        beaboid[1] = beaboid[1]-self.speed
        beaboid[2] = beaboid[2]/self.view 
        beaboid[3:] = beavector/self.view
        minboid[0] = mindrot
        minboid[1] = minboid[1]-self.speed
        minboid[2] = (minboid[2]-30)/self.view
        minboid[3:] = minvector/self.view
        
        control = self.ki.predict(np.concatenate((beaboid,minboid)).reshape(-1,1)).reshape(-1)
            
        self.rot += np.clip(control[0]-control[1],-0.2,0.2)
        self.speed += np.clip(control[2]-control[3],-0.2,0.2)
        #self.rot += np.clip(control[0]-control[1],-1,1)/5
        #self.speed += np.clip(control[2]-control[3],-1,1)/5
        self.speed = np.clip(self.speed,2,8)
        
        if minboid[2] < 1:
            if minboid[2] > 0:
                self.rating += ((1-beaboid[2])*5*self.count)**2
            else:
                self.rating -= (minboid[2]*100)**2
        else:
            self.rating = np.clip(self.rating,-999999,0)
        
        #self.rating += (aveboid[2]*10/self.count)**2
        

        #self.rating += self.count * np.sign(-aveboid[4])

        
    
class Flock(object):
    def __init__(self,field = None,size = 10):
        self.field = field
        self.size = size
        
    def create_swarm(self,target):
        field_size = self.field.get_size()
        self.boids = []
        for x in range(self.size):
            init_rot = np.random.random()*2*np.pi
            init_x = field_size[0]*np.random.random()
            init_y = field_size[1]*np.random.random()
            init_speed = 5*np.random.random()+2
            ki = Sequential()
            ki.init_network(target[0],target[1],target[2])
            self.boids.append(Boid(x,self.field,self.boids,ki,init_rot,(init_x,init_y),init_speed,300))
    
    def update(self):
        for boid in self.boids:
            boid.update()
            
    def rating(self):
        average_rating = 0
        for boid in self.boids:
            average_rating += boid.rating
        return average_rating/self.size
    
def boid_wrapper(targetids,targets,cache):
    field = Field()
    myflock = Flock(field,10)
    ratings = []
    for targetid, target in zip(targetids,targets):
        network = vec2net(target)
        myflock.create_swarm(network)
        for tick in range(600):
            myflock.update()
        ratings.append([targetid,myflock.rating()])
    with open(cache+".cache",'wb+') as handle:
        pickle.dump(ratings,handle,protocol = pickle.HIGHEST_PROTOCOL)
        handle.close
        
        
if __name__ == '__main__':
    freeze_support()
    evolution = Evolution((10,4),200,50,0.25,1,0.5,0.0,0.0,0.5,0.1)
    #evolution.load_model('boids')
    
    years = 20 # number of generation
    number_of_cores = 6 # number of processor core
    
    for year in range(years):
        print('Running Year: '+str(year+1))
        boidsRunning = []
        for x in range(number_of_cores):
            print("Starting Worker: " + str(x))
            targets = range(x,evolution.get_population_size(),number_of_cores)
            #boid_wrapper(np.array(targets),evolution.get_target(targets),"cache/boid_"+str(x))
            boidP = Process(target = boid_wrapper , args = (np.array(targets),evolution.get_target(targets),"cache/boid_"+str(x)))
            boidsRunning.append(boidP)
            boidP.start()
            # Extending the task to multicoreprocessing
        for boidP in boidsRunning:
            boidP.join()
            print("A Process finished.")
        ratings = [] 
        for x in range(number_of_cores):
            with open("cache/boid_"+str(x)+".cache",'rb') as handle:
                ratings.extend(pickle.load(handle))
                handle.close()
        for result in ratings:
            evolution.reward_target(result[1],result[0])
        fittest = evolution.get_fittest_network()
        replayBoid = evolution.get_target([fittest])
        with open("replay/boid_{:03d}.mdl".format(evolution.evolution),'wb+') as handle:
            pickle.dump(replayBoid,handle,protocol = pickle.HIGHEST_PROTOCOL)
            handle.close()
        evolution.envolve()
        evolution.save_model('boids')