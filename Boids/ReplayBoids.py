# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:48:00 2020

@author: phil-
"""
from Boids import *
import pygame
import numpy as np
from Genetics import *
from glob import glob
import pickle

def standard():
    pygame.init()
    field = Field()
    gameDisplay = pygame.display.set_mode(field.get_size())
    clock = pygame.time.Clock()
    boid_files = ['driver/swarm.mdl'] * 10 #list(glob('replay/*.mdl'))
    boid_files.sort()
    for year,boid_file in enumerate(boid_files):
        with open(boid_file,'rb') as handle:
            boid = pickle.load(handle)
            print(boid_file)
            print("Running best Boids out of year: "+ str(year+1))
        myflock = Flock(gameDisplay,field,50)
        target = vec2net(boid[0])
        myflock.create_swarm(target)
        ticks = 0
        while ticks < 300:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
            gameDisplay.fill((0,0,0))
            myflock.update()
            clock.tick(30)
            ticks += 1
            pygame.display.update()
    
def fieldmode():
    pygame.init()
    field = Field(size = (1000,1000))
    gameDisplay = pygame.display.set_mode(field.get_size())
    clock = pygame.time.Clock()
    swarmfile =  'driver/swarm.mdl'
    fieldfile =  'driver/field.mdl'
    ciclefile =  'driver/circle.mdl'
    ciclefile2 =  'driver/circle2.mdl'
    with open(swarmfile,'rb') as handle:
        swarmmdl = pickle.load(handle)
        swarmnet = vec2net(swarmmdl[0])
    with open(fieldfile,'rb') as handle:
        fieldmdl = pickle.load(handle)
        fieldnet = vec2net(fieldmdl[0])
    with open(ciclefile,'rb') as handle:
        circlemdl = pickle.load(handle)
        circlenet = vec2net(circlemdl[0])
    with open(ciclefile2,'rb') as handle:
        circlemdl2 = pickle.load(handle)
        circlenet2 = vec2net(circlemdl2[0])

    myflock = Flock(gameDisplay,field,50)
    myflock.create_swarm(swarmnet)
    ticks = 0
    drivemode = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
        gameDisplay.fill((0,0,0))
        myflock.update()
        clock.tick(60)
        if ticks > 300:
            ticks = 0
            if drivemode == 0:
                myflock.update_swarm(fieldnet,green)
                drivemode = 1
            elif drivemode == 1:
                myflock.update_swarm(swarmnet,white)
                drivemode = 2
            elif drivemode == 2:
                myflock.update_swarm(circlenet,red)
                drivemode = 3
            elif drivemode == 3:
                myflock.update_swarm(circlenet2,blue)
                drivemode = 0
        ticks += 1
        pygame.display.update()

if __name__ == '__main__':
    fieldmode()
    #standard()