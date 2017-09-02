import tensorflow as tf
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
NX = 60 
NY = 60 

def neural_network_model(input_size):
    network = input_data(shape = [None,NY,NX//3],name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 1, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    
    X = [i[1] for i in training_data]
    Y = np.array([i[0] for i in training_data]).reshape(size,1)
    
    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input':X}, {'target':Y}, n_epoch=5, snapshot_step=500, show_metric=True,
              run_id='openaistuff')

    return model

with open("BD", "r") as f:
            training_data = []
            size=0
            for line in f.readlines():
                size+=1
                point,row = line.strip().split(":")
                stat00,foo = row.split(";")
                stat0 = [[0 for x in range(NX//3)] for y in range(NY)] 
                for y in range(NY):
                    for x in range(NX//3):
                        value = stat00[((y+x+1)*3)-1]
                        stat0[y][x] = int(value)
                training_data.append([int(point),stat0])

model = train_model(training_data)

Ran_board = np.array(training_data[0][1]).reshape(1,NY,NX//3)
print(np.argmax(model.predict(Ran_board)))
model.save('COGL.model')
for game in range(1):
    Ran_board = np.array([[ random.randint(0, 1) for x in range(NX//3)] for y in range(NY)]).reshape(1,NY,NX//3)
    print(np.argmax(model.predict(Ran_board)))
                                                      
