from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import PIL 
import PIL.ImageOps
import os
from NeuralNetwork import NeuralNetwork
        
class Painting(object):
    def __init__(self):
        self.neuron = NeuralNetwork([64,16,16,10])
        self.neuron.load_model('digits')
        pass
    
    def predict(self):
        # save postscipt image 
        self.canvas.postscript(file ='pic.ps') 
        # use PIL to convert to PNG 
        img = PIL.Image.open('pic.ps').convert('L')
        img = img.resize((8, 8), PIL.Image.ANTIALIAS)
        img = PIL.ImageOps.invert(img)
        target = np.array(img)
        prediction = np.argmax(self.neuron.predict(target.flatten().reshape(-1,1)))
        self.label['text'] = "Das ist eine " + str(prediction)
        os.remove('pic.ps')
        print(prediction)

        
    def paint(self,event):
    	python_green = "#476042"
    	x1, y1 = ( event.x ), ( event.y )
    	self.canvas.create_oval( x1-5, y1-5, x1+5, y1+5, fill = python_green ,width = 10)
        
    def clear(self):
        self.canvas.delete('all')
        #self.canvas.create_rectangle(0+3,0+3,8*32-3,8*32-3,width = 2)
    
    def construct(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=8*32, height=8*32)
        self.label = tk.Label(self.root, text = "Das ist keine Zahl",font = ("Arial",18))
        self.button = tk.Button(self.root, text = "Predict",command = self.predict)
        self.button_2 = tk.Button(self.root, text = "Clear",command = self.clear)
        self.button_2.pack()
        self.button.pack()
        self.canvas.pack()
        self.canvas.bind( "<B1-Motion>", self.paint )
        self.label.pack()
        self.root.mainloop()
    	         
def vectorize(index):
    _temp = np.zeros((10,1))
    _temp[index] = 1
    return _temp

train = False

if train:
    neuron = NeuralNetwork([64,16,16,10])
    neuron.load_model('digits')
    digits = load_digits()     
    x = digits['data']
    
    y = [vectorize(j) for j in digits['target']]
    training_data = list(zip(x,y))
    neuron.train(training_data[:-100],10,100,0.1)#,training_data[-100:])
    #neuron.save_model("digits")
    target = 0
    while target != -1:
        target = int(np.random.randint(0,len(training_data)))
        prediction = np.argmax(neuron.predict(x[target].reshape(-1,1)))
        truth = digits['target'][target]
        
        plt.imshow(digits['images'][target])
        plt.title('Prediction: ' + str(prediction) + '  // Truth: ' + str(truth))
        plt.draw()
      
        plt.pause(1)
        plt.clf()
else:
    myNumber = Painting()
    myNumber.construct()
    