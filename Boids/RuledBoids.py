# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:06:34 2020

@author: phil-
"""
import pygame
import numpy as np

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)


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
            pos[0] = self.size[0]
        if pos[1] < 0:
            pos[1] = self.size[1]
        if pos[0] > self.size[0] :
            pos[0] = 0
        if pos[1] > self.size[1] :
            pos[1] = 0
        return pos
        
    def get_size(self):
        return self.size
        
        

class Triangle(object):
    def __init__(self,screen = None,color = (0,0,0),size = (1,1)):
        self.color = color
        self.screen = screen
        self.points = ((size[0],0),(0,size[1]*0.5),(0,-size[1]*0.5))
        self.cache = np.zeros((3,2))
    
    def draw(self,rotm,pos):
        for idx,point in enumerate(self.points):
            m = np.dot(rotm, point)
            self.cache[idx] = m + pos
        pygame.draw.polygon(self.screen,self.color,self.cache)
     
class Boid(object):
    def __init__(self,idx = 0,screen = None,field = None,others = None,init_rot = 0,init_pos = (0,0),init_speed = 1,view = 100):
        self.rot = init_rot 
        self.pos = np.array(init_pos,dtype=np.float16)
        self.speed = init_speed
        self.form = Triangle(screen,white,(20,10))
        self.field = field
        self.others = others
        self.view = view
        self.cache = np.zeros((50,5))
        self.count = 0
        self.idx = idx
        self.screen = screen
        
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
        # else:
        #     self.rot += drot
        
        self.pos = self.field.get_position(self.pos)
        self.flocking(irotm)
            
        if self.rot < 0:
            self.rot = 2*np.pi
        elif self.rot >= 2*np.pi:
            self.rot = 0
            
        self.form.draw(rotm,self.pos)
    
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
        aveboid = np.sum(self.cache[0:self.count],axis=0)/self.count
        min_dist = self.cache[0:self.count,2]
        minboid = self.cache[np.argmin(min_dist)]
        vector = np.asarray(np.dot(irotm,aveboid[3:]-self.pos)).reshape(-1)
        minvector = np.asarray(np.dot(irotm,minboid[3:]-self.pos)).reshape(-1)
        
        drot = aveboid[0]-self.rot
        if drot > np.pi:
            drot = -2*np.pi+drot
            
        # Place NN here !!!
            
            
        # alignment
        self.rot += np.clip(drot,-1,1) *0.3*(self.view-aveboid[2])/(self.view)
        self.speed +=  np.clip((aveboid[1]-self.speed),-1,1)*0.1*(self.view-aveboid[2])/(self.view)
        
        # cohesion
        self.speed += np.clip(aveboid[2]/self.view,-0.05,0.05) * np.sign(vector[0])
        self.rot -= np.clip(aveboid[2]/self.view * -np.sign((vector[0]+50)*vector[1]),-0.01,0.01)
        
        # separation
        if minboid[2] < 50:
            self.speed += np.clip((minboid[2]-50)/30*np.sign(minvector[0]),-0.1,0.1)
            self.rot += np.clip((minboid[2]-50)/30*np.sign(minvector[1]),-0.05,0.05) * np.sign(minvector[0]+50) 
        
        self.speed = np.clip(self.speed,3,7)
        
        
        
    
class Flock(object):
    def __init__(self,screen = None,field = None,size = 10):
        self.field = field
        self.size = size
        self.boids = []
        self.screen = screen
        
    def create_swarm(self):
        field_size = self.field.get_size()
        for x in range(self.size):
            init_rot = np.random.random()*2*np.pi
            init_x = field_size[0]*np.random.random()
            init_y = field_size[1]*np.random.random()
            init_speed = 5*np.random.random()+2
            self.boids.append(Boid(x,self.screen,self.field,self.boids,init_rot,(init_x,init_y),init_speed,300))
    
    def update(self):
        for boid in self.boids:
            boid.update()
        
if __name__ == '__main__':
    pygame.init()
    field = Field()
    gameDisplay = pygame.display.set_mode(field.get_size())
    clock = pygame.time.Clock()
    myflock = Flock(gameDisplay,field,150)
    myflock.create_swarm()
    ticks = 0
    while True:
        if ticks >= 300:
            myflock = Flock(gameDisplay,field,150)
            myflock.create_swarm()
            ticks = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
        gameDisplay.fill((0,0,0))
        myflock.update()
        clock.tick(120)
        ticks += 1
        pygame.display.update()