# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:06:34 2020

@author: phil-
"""
import pygame
import numpy as np
from Genetics import *


black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

class Field(object):
    def __init__(self,size = (1000,1000),limit = 150):
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
    
    def get_position(self,pos):
        if pos[0] < 0:
            pos[0] = self.size[0]-1
        if pos[1] < 0:
            pos[1] = self.size[1]-1
        if pos[0] > self.size[0] :
            pos[0] = 1
        if pos[1] > self.size[1] :
            pos[1] = 1
        return pos
        
    def get_size(self):
        return self.size
        
        

class Triangle(object):
    def __init__(self,screen = None,color = (0,0,0),size = (1,1)):
        self.color = np.array(color)
        self.screen = screen
        self.points = ((size[0],0),(0,size[1]*0.5),(0,-size[1]*0.5))
        self.cache = np.zeros((3,2))
    
    def draw(self,rotm,pos,rating):
        for idx,point in enumerate(self.points):
            m = np.dot(rotm, point)
            self.cache[idx] = m + pos
        pygame.draw.polygon(self.screen,self.color,self.cache)
     
class Boid(object):
    def __init__(self,idx = 0,screen = None,field = None,others = None,ki = None,init_rot = 0,init_pos = (0,0),init_speed = 1,view = 200):
        self.rot = init_rot 
        self.pos = np.array(init_pos,dtype=np.float16)
        self.speed = init_speed
        self.form = Triangle(screen,(255,255,255),(20,10))
        self.field = field
        self.others = others
        self.view = view
        self.cache = np.zeros((50,5))
        self.count = 0
        self.idx = idx
        self.screen = screen
        self.ki = ki
        self.rating = 0
        
    def update(self):
        c,s = np.cos(self.rot), np.sin(self.rot)
        rotm = np.matrix([[c, -s], [s, c]])
        irotm = np.matrix([[c, s], [-s, c]])
        dirvec = np.asarray(np.dot(rotm,[1,0])).reshape(-1)
        self.pos[0] += self.speed*c
        self.pos[1] += self.speed*s
        self.get_nearest()
        
        # drot = self.field.get_direction(dirvec,self.pos[0],self.pos[1])*0.005 *self.speed
        # if drot == 0:
        #     self.flocking(irotm)
        #     pass
        # else:
        #     self.rot += drot
        
        self.pos = self.field.get_position(self.pos)
        self.flocking(irotm)
            
        if self.rot < 0:
            self.rot = 2*np.pi
        elif self.rot >= 2*np.pi:
            self.rot = 0
            
        self.form.draw(rotm,self.pos,self.rating)
    
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
        minboid[2] = (minboid[2]-50)/self.view
        minboid[3:] = minvector/self.view
        
        control = self.ki.predict(np.concatenate([beaboid,minboid]).reshape(-1,1)).reshape(-1)
            
        self.rot += np.clip(control[0]-control[1],-0.2,0.2)
        self.speed += np.clip(control[2]-control[3],-0.2,0.2)
        #self.rot += np.clip(control[0]-control[1],-1,1)/3
        #self.speed += np.clip(control[2]-control[3],-1,1)/3
        self.speed = np.clip(self.speed,3,8)
    
class Flock(object):
    def __init__(self,screen = None,field = None,size = 10):
        self.field = field
        self.size = size        
        self.screen = screen

        
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
            self.boids.append(Boid(x,self.screen,self.field,self.boids,ki,init_rot,(init_x,init_y),init_speed,300))
            
    def update_swarm(self,target,color):
        for boid in self.boids:
            boid.ki.init_network(target[0],target[1],target[2])
            boid.form.color = color
    
    def update(self):
        for boid in self.boids:
            boid.update()
            