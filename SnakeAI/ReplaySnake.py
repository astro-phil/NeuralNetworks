# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:02:11 2019

@author: phil-
"""
import pygame
import pickle
import numpy as np
import math
import random
from NeuralNetwork import NeuralNetwork
import time

class ReplaySnake(object):
    '''
    This is a snake game used as environment for the neural network.
    '''
    def __init__(self,height = 400,width = 400,block = 10,ANN=NeuralNetwork([24,18,18,4])):
        self.height = height
        self.width = width
        self.block = block
        self.ANN = ANN       
        pygame.init()
        self.display = pygame.display.set_mode((height, width))
        pygame.display.set_caption('Snake Game by Edureka')
        self.clock = pygame.time.Clock()
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("comicsansms", 35)
        self.white = (255, 255, 255)
        self.yellow = (255, 255, 102)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)     
        
    def foodCollide(self,pos,food):
        # checks if there is food
        if pos[0] == food[0] and pos[1] == food[1]:
            return True
        return False
    
    def bodyCollide(self,pos,body):
        # checks if there is the snake
        for part in body[:-1]:
            if pos[0] == part[0] and pos[1] == part[1]:
                return True
        return False
        
    def wallCollide(self,pos):
        # checks if there is the wall
        if(pos[0] >= self.height-(self.block) or pos[0] < self.block or pos[1] >= self.height-(self.block) or pos[1] < self.block):
            return True
        return False
            
    def lookInDirection(self,direction,head,food,body):
        # looks into the given direction from the snakes head
        look = np.array([0.0,0.0,0.0])
        distance = 0.0
        pos = np.array(head)
        foundBody = False
        pos += direction
        distance +=1
        while (not self.wallCollide(pos)):
            if(self.foodCollide(pos,food)):
                look[0] = 1
            if(not foundBody and self.bodyCollide(pos,body)):
                look[1] = 1.0/distance
                foundBody = True
            pos += direction
            distance += 1
        look[2] = 1.0/distance
        return look
        
    def snake_look(self,head,food,body):
        # The snake looks in 8 direction for distance to wall,itself and for food
        directions = [[-self.block,0],[-self.block,-self.block],[0,-self.block],[self.block,-self.block],[self.block,0],[self.block,self.block],[0,self.block],[-self.block,self.block]]
        data = []
        for direction in directions:
            data.extend(self.lookInDirection(np.array(direction),head,food,body))
        return np.array(data)
               
    def fitness(self,score,lifetime):
        # the calcualtion of the fitness / score of the snakes run
        if(score < 10):
            fitness = math.floor(lifetime * lifetime) * math.pow(2,score);
        else:
            fitness = math.floor(lifetime * lifetime)
            fitness *= math.pow(2,10)
            fitness *= (score-9)
        return fitness
    
    def display_score(self,score):
        value = self.score_font.render("Your Score: " + str(score), True, self.yellow)
        self.display.blit(value, [0, 0])
     
    def draw_snake(self, snake_list):
        for x in snake_list:
            pygame.draw.rect(self.display, self.black, [x[0], x[1], self.block, self.block])

    def run(self,food_list,netvector):
        # the gamesmainloop
        game_over = False
        x1 = self.width/2
        y1 = self.height/2
        x1_change = 0
        y1_change = 0
        weights,biases = self.ANN.vec2net(netvector,self.ANN.sizes)
        self.ANN.init_network(weights,biases)
        snake_List = []
        Length_of_snake = 5
        food_idx = 0
        food = food_list[food_idx]
        
        direction = 1
        ticks = 200
        score = 1
        lifetime = 0
        while not game_over:                                    
            if len(snake_List)>0:        
                data = self.snake_look([x1,y1],food,snake_List)
                prediction = self.ANN.predict(data.reshape(-1,1),function = self.ANN.relu)
                # the neural network predicts the direction
                direction = np.argmax(prediction)
            
            if direction == 0:
                x1_change = 0
                y1_change = -self.block
            elif direction == 1:
                x1_change = 0
                y1_change = self.block
            elif direction == 2:
                y1_change = 0
                x1_change = -self.block
            elif direction == 3:
                y1_change = 0
                x1_change = self.block
            
            snake_Head = [x1,y1]
            snake_List.append(snake_Head)
            
            if len(snake_List) > Length_of_snake:
                del snake_List[0]
            
            if self.foodCollide(snake_Head,food):
                food_idx += 1
                if food_idx < len(food_list):
                    food = food_list[food_idx]
                else:
                    food = [round(random.randrange(2*self.block, self.width - 2*self.block) / 10.0) * 10.0, round(random.randrange(2*self.block, self.height - 2*self.block) / 10.0) * 10.0]
                Length_of_snake += 10
                score += 1
                if(ticks < 300):
                    ticks += 200
            
            if (self.bodyCollide(snake_Head,snake_List) and len(snake_List)>1) or self.wallCollide(snake_Head) or ticks <= 0:
                game_over = True
        
            x1 += x1_change
            y1 += y1_change
            
            pygame.event.get()
            self.display.fill(self.blue)
            pygame.draw.rect(self.display, self.green, [food[0],food[1], self.block, self.block])
            self.draw_snake(snake_List)
            self.display_score(score)
            pygame.display.update()
            self.clock.tick(50)
            ticks -= 1
            lifetime += 1
            
if __name__ == '__main__':
    ReplaySnakes = []
    with open("replay.snake",'rb') as handle:
        ReplaySnakes = pickle.load(handle)
        handle.close()
        # loads the best snakes from every generation of the evolution 
    #time.sleep(10)
    SnakeTemplate = ReplaySnake()
    for year,settings in enumerate(ReplaySnakes):
        print("Running best Snake out of year: "+ str(year+1))
        SnakeTemplate.run(settings[1],settings[0])
        # run the specified snake
    #settings = ReplaySnakes[-2]
    #SnakeTemplate.run(settings[1],settings[0])
    pygame.quit() 

    