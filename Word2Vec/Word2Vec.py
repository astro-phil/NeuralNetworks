# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:52:43 2019

@author: phil-
"""
import numpy as np
import re
import os
import pickle
from scipy.spatial import distance

class Tokensizer(object):
    '''
    This class contains all the needed functions to extract word from a given textfile or text.
    To exctract userdefind tokens cahnge the re.compile()
    '''
    def __init__(self):
        self.tokens = []
        self.count = dict()
        self.word_to_id = dict()
        self.id_to_word = dict()
        self.pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        
    def extract_tokens(self,text):
        # This functions extracts all tokens and their occurence from a string  
        self.tokens = self.pattern.findall(text.lower())
        # This line splits the text into a list of words and removes all non-word entries
        for token in self.tokens:
            if token in self.count.keys():
                self.count[token] += 1
            else:
                self.count[token] = 1
        print("Successfully extracted " + str(len(self.count.keys())) + " tokens from the given text.")
    
    def extract_tokens_from_file(self,path):
        # This functions extracts all tokens and their occurence from a text file
        if not os.path.isfile(path):
            return None
        with open(path,'r',encoding='utf-8') as handle:
            text = handle.read()
            # Reads the entire text file at once and stores it into a string 
            handle.close()
        self.tokens = self.pattern.findall(text.lower())
        # This line splits the text into a list of words and removes all non-word entries
        for token in self.tokens:
            if token in self.count.keys():
                self.count[token] += 1
            else:
                self.count[token] = 1
        print("Successfully extracted " + str(len(self.count.keys())) + " tokens from the given text.")
        
    def clean_tokens(self,threshold = 100,whitelist = [],blacklist = []):
        # Use this function if you want to remove unecessary words from the trainingdata
        # A custom black and whitelist filled with string could be set 
        # Threshold is used to define how many word will be removed automaticly
        words = []
        occurence = []
        cleaned = blacklist
        for wo in self.count.items():
            words.append(wo[0])
            occurence.append(wo[1])
        idxs = np.argsort(occurence, axis=0)[-threshold:] 
        #Sorts the array and return the top threshold highest indexs 
        for idx in idxs:
            if not words[idx] in whitelist:
                cleaned.append(words[idx])
        for word in cleaned:
            while word in self.tokens:
                self.tokens.remove(word)
        # These to loops remove the unwanted words from the text
        print('The following tokens have been removed from the given text : \n',cleaned)
        return cleaned
        
    def map_tokens_to_id(self):   
        # This function generates a mapping from id to word and word to id
        # This is used to reduce memory consumption instead of a one-hot-vector
        for i, token in enumerate(set(self.tokens)):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        return self.tokens
    
    def get_mapping(self):
        return self.word_to_id, self.id_to_word
    
    def get_vocab_size(self):
        return len(self.word_to_id.keys())
    
    def generate_training_data(self, window_size):
        # This function generates the Trainingdata from the text
        # it uses a targetword as X and generates Y out of the surrounding words with a given windowsize
        N = len(self.tokens)
        X, Y = [], []
        for i in range(N):
            nbr_inds = list(range(max(0, i - window_size), i)) + \
                       list(range(i + 1, min(N, i + window_size + 1)))
            for j in nbr_inds:
                X.append(self.word_to_id[self.tokens[i]])
                Y.append(self.word_to_id[self.tokens[j]])
        X = np.array(X)
        X = np.expand_dims(X, axis=0)
        Y = np.array(Y)
        Y = np.expand_dims(Y, axis=0)
        return X,Y
    
    def save_mapping(self,name = "tokens"):
        # Saves the id to word and the word to id dictionary into a file
        data = [self.word_to_id,self.id_to_word]
        with open(name+".map",'wb+') as handle:
            pickle.dump(data,handle)
            handle.close
    
    def load_mapping(self,name = "tokens"):
        # loads the id to word and the word to id dictionary from a file
        with open(name+".map",'rb') as handle:
            data = pickle.load(handle)
            handle.close
        self.word_to_id = data[0]
        self.id_to_word = data[1]


class Word2Vec(object):
    ''' 
    This is the Word2Vec Embedding object
    is uses the skipgram or cbow method to generate vector representation for words
    the emb_size is the dimension of the word vectors
    '''
    
    def __init__(self,emb_size = 100, vocab_size = 100,learning_rate=0.01 ,id_to_word=dict()):
        self.emb_size = emb_size
        self.vocab_size = vocab_size 
        self.word_emb = np.random.randn(vocab_size,emb_size) *1
        self.weights = np.random.randn(vocab_size,emb_size) *1
        self.learning_rate = learning_rate
        self.id_to_word = id_to_word
        
    def idx_to_vec(self,word_idx):
        # returns the wordvector by thze given indexes
        word_vec = self.word_emb[word_idx.flatten(), :].T
        return word_vec
        
    def softmax(self,Z):
        # the softmax activation function 
        activation = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True))    
        return activation
    
    def delta_cross_entropy(self,Z,word_idx):
        # the cost and the derivative of the softmax function combined to the delta cross entropy
        m = word_idx.shape[1]
        dL_dZ = np.array(Z)
        dL_dZ[word_idx, np.arange(m)] -= 1.0
        return dL_dZ
        
    def cross_entropy(self,Z, word_idx):
        # cost function calculation the score of the prediction
        m = Z.shape[1]
        cost = -(1 / m) * np.sum(np.log(Z[word_idx, np.arange(m)]))
        return cost
    
    def forward(self,word_idx):
        # forward propagation / prediction of the target words
        word_vec = self.idx_to_vec(word_idx)
        Z = np.dot(self.weights,word_vec)
        activation = self.softmax(Z)
        return activation
    
    def nearest(self,word_vec):
        # prediction of the nearest words to a given target word
        Z = np.dot(self.weights,word_vec)
        activation = self.softmax(Z)
        return activation
    
    def linear_reverse(self,dL_dZ,word_vec):
        # linear multiplication in reverse to calculate the changes in the higher layer
        dL_dW = (1 / word_vec.shape[1]) * np.dot(dL_dZ, word_vec.T)
        dL_dword = np.dot(self.weights.T, dL_dZ)
        return dL_dW, dL_dword
    
    def backward(self,activation,word_idx,Y):
        # the actual training of the neural net and calculation of the changes to the Layerparameters
        word_vec = self.idx_to_vec(word_idx)
        dL_dZ = self.delta_cross_entropy(activation,Y)
        dL_dW, dL_dword = self.linear_reverse(dL_dZ,word_vec)
        return dL_dW, dL_dword
        
    def update_parameters(self,word_idx,dL_dW,dL_dword):
        # updating the layerparameters
        self.weights -= self.learning_rate * dL_dW
        self.word_emb[word_idx, :] -= dL_dword.T * self.learning_rate
        
    def skipgram_training(self,X,Y,batch_size=256,epochs = 100):
        # Skipgram training method with the use of minibatches and Stochastic Gradient Decent
        # This is still neural network only without offset and hiddenlayers
        # batch_size is used to define how many smaples the network should use at once for training
        # epochs is used to define how many over all interation the network should proceed
        m = X.shape[1]
        for epoch in range(epochs):
            epoch_cost = 0
            batch_inds = list(range(0, m, batch_size))
            np.random.shuffle(batch_inds)
            for i in batch_inds:
                # iterating through the minibatches
                X_batch = X[:, i:i+batch_size]
                Y_batch = Y[:, i:i+batch_size]
                activation = self.forward(X_batch)
                dW,dword = self.backward(activation, X_batch,Y_batch)
                self.update_parameters(X_batch,dW,dword)
                cost = self.cross_entropy(activation, Y_batch)
                epoch_cost += np.squeeze(cost) 
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            if epoch % (epochs // 100) == 0:
                #self.learning_rate *= 0.98
                self.save_model()
                # saving the actual state of the neural network
                
    def cbow_training(self,X,Y,batch_size=128,epochs = 100):
        # CBOW training method with the use of minibatches and Stochastic Gradient Decent
        # This is still neural network only without offset and hiddenlayers
        # batch_size is used to define how many smaples the network should use at once for training
        # epochs is used to define how many over all interation the network should proceed
        m = X.shape[1]
        for epoch in range(epochs):
            epoch_cost = 0
            batch_inds = list(range(0, m, batch_size))
            np.random.shuffle(batch_inds)
            for i in batch_inds:
                # iterating through the minibatches
                Y_batch = X[:, i:i+batch_size]
                X_batch = Y[:, i:i+batch_size]
                activation = self.forward(X_batch)
                dW,dword = self.backward(activation, X_batch,Y_batch)
                self.update_parameters(X_batch,dW,dword)
                cost = self.cross_entropy(activation, Y_batch)
                epoch_cost += np.squeeze(cost) 
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            if epoch % (epochs // 100) == 0:
                #self.learning_rate *= 0.98
                self.save_model()
                # saving the actual state of the neural network
                
    def save_model(self,mdl_name = "wordembeddings"):
        # this function saves the actual state of the neural network to a file
        print("Saving Embeddings ..")
        data = [self.word_emb,self.weights,self.learning_rate]
        with open(mdl_name+".cache",'wb+') as handle:
            pickle.dump(data,handle,protocol = pickle.HIGHEST_PROTOCOL)
            handle.close
            
    def load_model(self,mdl_name = "wordembeddings"):
        # this function loads the actual state of the neural network from a file
        with open(mdl_name+".cache",'rb') as handle:
            data = pickle.load(handle)
            handle.close
        self.weights = data[1]
        self.word_emb = data[0]
        #self.learning_rate = data[2]
    
    def get_word_embedding(self):
        # return a dictionary with all the word vectors
        embeddings = dict()
        for idx,vector in enumerate(self.weights):
            embeddings[self.id_to_word[idx]] = vector
        return embeddings

def closest_node_square(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argsort(dist_2, axis=0)[-4:]

def closest_node_cosine(target, nodes):
    nodes = np.array(nodes)
    dist = np.zeros(len(nodes))
    for idx,node in enumerate(nodes):
        dist[idx] = distance.cosine(target,node)
    return np.argsort(dist, axis=0)[-4:]

parser = Tokensizer()

#doc = "Kayron sitzt gemütlich an einem leeren Tisch, irgendwo fern von Musik, Spiel und Tanz. Durch den Lunar-Day sammelt sich der Trubel im Cryo vor der Bar. Viel Trank für kleines Geld. Das Cryo ist selten so voll wie an diesem Tag. Jede Art von Persönlichkeit hat es hierher verschlagen, vom Obersten Rat bis zum einfachen Arbeiter. Alle haben sich hier versammelt. Selbst Iris, der Raumstation, merkt man an, dass dies ein besondere Tag ist. Die ganze Außenbeleuchtung, die sonst das ganze Jahr ausgeschaltet ist, zeigt heute ihr volles Potential. Vor 5 Jahren endete die Rebellion mit der Unterzeichnung des Friedensabkommens auf Eos und einem der größten Blutmonde seit hunderten von Jahren. Daher auch der Name Lunar-Day. In der Zwischenzeit hat sich Kayron einen Drink besorgt, sitzt wieder in Gedanken vertieft an seinem Tisch und starrt zum Fenster hinaus. Die Beleuchtung von Iris fügt sich so wunderbar in den Sternenhintergrund ein, dass man kaum mehr einen Unterschied erkennt. Kayron verfällt in Gedanken an die Vergangenheit."

#tokens = parser.extract_tokens(doc)
#tokens = parser.extract_tokens_from_file('D:/MyCloud/Coding/Python/Neuronales Netz/Test.txt')
blacklist = ['ob','also','ab']
whitelist = ['lennard','kayron','eyreen','marc','endor','gideon','iris','schiff','tobias']
parser.load_mapping()

parser.clean_tokens(whitelist=whitelist,blacklist=blacklist)
#parser.map_tokens_to_id()
vocab_size = parser.get_vocab_size()
word_to_id, id_to_word = parser.get_mapping()
#X, Y = parser.generate_training_data(5)
#parser.save_mapping()


WordTraining = Word2Vec(emb_size = 300 , vocab_size = vocab_size,learning_rate=0.05,id_to_word = id_to_word)
WordTraining.load_model()
#WordTraining.skipgram_training(X,Y,batch_size = 128,epochs = 2000)
#WordTraining.cbow_training(X,Y,batch_size = 128,epochs = 2000)

embeddings = WordTraining.get_word_embedding()
print("Finish")


#X_test = np.arange(vocab_size)
#X_test = np.expand_dims(X_test, axis=0)
#softmax_test = WordTraining.forward(X_test)
#top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]
#
#for input_ind in range(vocab_size):
#    input_word = id_to_word[input_ind]
#    output_words = [id_to_word[output_ind] for output_ind in top_sorted_inds[::-1, input_ind]]
#    print("{}'s neighbor words: {}".format(input_word, output_words))

target_vec = embeddings['marc']-embeddings['kayron']+embeddings['eyreen']
softmax_test = WordTraining.nearest(target_vec)
top_sorted_inds = np.argsort(softmax_test, axis=0)[-5:]
output_words = [id_to_word[output_ind] for output_ind in top_sorted_inds[:-1]]
print(output_words)

#words = []
#vectors = []
#for ab in embeddings.items():
#    words.append(ab[0])
#    vectors.append(ab[1])
#    
#top_sorted_inds = closest_node_cosine(embeddings['kayron'],vectors)
#output_words = [words[output_ind] for output_ind in top_sorted_inds[:-1]]
#print(output_words)
#
#
#top_sorted_inds = closest_node_square(target_vec,vectors)
#output_words = [words[output_ind] for output_ind in top_sorted_inds[:-1]]
#print(output_words)


        