
import random
import pickle
import numpy as np
import math
from NeuralNetwork import NeuralNetwork
from multiprocessing import Process,freeze_support
    
class SilentSnake(object):
    '''
    This is a snake game used as environment for the neural network.
    '''
    def __init__(self,height = 400,width = 400,block = 10,ANN=NeuralNetwork([24,18,18,4])):
        # ANN is a container for the neural network
        self.height = height
        self.width = width
        self.block = block
        self.ANN = ANN        
        
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
    
    def runLoop(self,targets,netvectors,cache):
        # the gamesmainloop
        snakes = []
        for target,netvector in zip(targets,netvectors):
            game_over = False
            x1 = self.width/2
            y1 = self.height/2
            
            x1_change = 0
            y1_change = 0
            weights,biases = self.ANN.vec2net(netvector,self.ANN.sizes)
            self.ANN.init_network(weights,biases)
            snake_List = []
            Length_of_snake = 5
            food_list = []
            food = [round(random.randrange(2*self.block, self.width - 2*self.block) / 10.0) * 10.0, round(random.randrange(2*self.block, self.height - 2*self.block) / 10.0) * 10.0]
            food_list.append(food)
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
                    food = [round(random.randrange(2*self.block, self.width - 2*self.block) / 10.0) * 10.0, round(random.randrange(2*self.block, self.height - 2*self.block) / 10.0) * 10.0]
                    food_list.append(food)
                    Length_of_snake += 10
                    score += 1
                    if(ticks < 300):
                        ticks += 200
                        
                if (self.bodyCollide(snake_Head,snake_List) and len(snake_List)>1) or self.wallCollide(snake_Head) or ticks <= 0:
                    game_over = True
            
                x1 += x1_change
                y1 += y1_change
                
                ticks -= 1
                lifetime += 1
            snakes.append([target,self.fitness(score,lifetime),food_list])
        with open(cache+".cache",'wb') as handle:
            pickle.dump(snakes,handle,protocol = pickle.HIGHEST_PROTOCOL)
            handle.close

if __name__ == '__main__':
    freeze_support()
    evolution = NeuralNetwork([24,18,18,4])
    evolution.init_genetics()
    load = False
    SnakeTemplate = SilentSnake()
    years = 40 # number of generation
    number_of_cores = 4 # number of processor core
    ReplaySnakes = []
    if load:
        evolution.load_model("snake")
        with open("replay.snake",'rb') as handle:
            ReplaySnakes = pickle.load(handle)
            handle.close()
        # loading the population and the snake replays 
    for year in range(years):  
        snakesRunning = []
        chunk_size = evolution.get_population_size()
        print('Running Year: '+str(year+1))
        for x in range(number_of_cores):
            print("Starting Worker: " + str(x))
            targets = range(x,chunk_size-3+x,number_of_cores)
            snake = Process(target = SnakeTemplate.runLoop , args = (np.array(targets),evolution.get_target(targets),"snake_"+str(x)))
            snakesRunning.append(snake)
            snake.start()
            # Extending the task to multicoreprocessing
        for snake in snakesRunning:
            snake.join()
            print("A Process finished.")
        # Waiting until all process are finished
        results = [] 
        food_lists = {}
        for x in range(number_of_cores):
            with open("snake_"+str(x)+".cache",'rb') as handle:
                results.extend(pickle.load(handle))
                handle.close()
        # Reading the multiprocessing caches 
        for result in results:
            evolution.reward_target(result[1],result[0])
            food_lists[str(result[0])] =  result[2]
        # Rewarding the neural networks
        fittest = evolution.get_fittest_network()
        ReplaySnakes.append([evolution.get_target(fittest),food_lists[str(fittest)]])
        evolution.save_model("snake")
        with open("replay.snake",'wb') as handle:
            pickle.dump(ReplaySnakes,handle,protocol = pickle.HIGHEST_PROTOCOL)
            handle.close()
        # Saving the best snakes for the replay
        evolution.envolve()
        