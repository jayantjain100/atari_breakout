#NOT WORKING

#Breakout - Jayants version
#search for 'LTR' - those are the tasks jo baad mei karne hain
# HARD, THINK
#follow the steps in the nb
#and khush ho ja bas

import gym
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Multiply
from keras.layers import Maximum
from matplotlib import pyplot as plt  

BUFFER_SIZE=1000
NUM_ACTIONS=4
#HARD: making it specific to breakout abhi ke liye

# class transition():
# 	self.

class cyclic_queue():
	def __init__(self, capacity):
		self.cap = capacity
		self.index=0
		self.array = np.zeros(capacity, dtype=tuple)
		#now it expects an object there- i need a better space optimisation, cause i know ki kitne size ki state ayegi, action , vagerah


	def add(self,k):
		self.array[self.index]= k
		self.index = (self.index+1)%self.cap

#funcs for image preprocessing-
#LTR: checkout ki jo frame return hota hai uska image kya hai and what part does it cover!!!

buff = cyclic_queue(BUFFER_SIZE)


model= Sequential()

#add all layers ab
#(layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#i took a screenshot of the game to actually see the dimensions and i was only able to reduce it to 208 x 180 =approx 210 x 160
#but i cant take a middle square out of this which contains all the useful info


model.add(Conv2D(16,(8,8) , strides=(4, 4),input_shape=(105, 80, 4), activation="relu"))
model.add(Conv2D(32, (4,4), strides=(2,2), activation="relu"))
#flatten?
# model.Flatten()
model.add(Dense(256,activation="relu"))
model.add(Dense(NUM_ACTIONS))
#THINK: here the activation was linear, AISA KYON
#filtering/masking layer
# model.add(Multiply())???????????????????????//variable mask//????????????????????????????????????????????????
model.compile(loss="mse", optimizer= "RMSprop", metrics=['accuracy'])
#LTR: metric kya dalun yahan bc?