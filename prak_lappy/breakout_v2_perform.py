#i have to make a functional model instead of a sequential one , for the complex structure
# i have two inputs na, ek toh 4image vala state hai and ek action mask hai, pehle ke liye convolution karna hai 
# and dusre ke liye bas end mei multiply

#Breakout - Jayants version
#search for 'LTR' - those are the tasks jo baad mei karne hain
# HARD, THINK
#follow the steps in the nb
#and khush ho ja bas

import gym
import numpy as np 
import sys
# from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import multiply
from keras.layers import Maximum
from keras.layers import Lambda
from matplotlib import pyplot as plt  
from keras import backend 
import scipy.misc
from PIL import Image


BUFFER_SIZE=300000
NUM_ACTIONS=4
FRAME_INPUT_SHAPE= (4,80,72) #(CHannels x Height x width)
ITERATIONS= 2000000
DECREASE_RATE= 250000
MINIBATCH_SIZE=32
GAMMA=0.99
#THINK: is this the correct sequence?- i guess cause frame return karke maine uska type dekha tha
#HARD: making it specific to breakout abhi ke liye

# class transition():
# 	self.

# class cyclic_queue():
# 	def __init__(self, capacity):
# 		self.cap = capacity
# 		self.index=0
# 		self.array = np.zeros(capacity, dtype=tuple)
# 		#now it expects an object there- i need a better space optimisation, cause i know ki kitne size ki state ayegi, action , vagerah


# 	def add(self,k):
# 		self.array[self.index]= k
# 		self.index = (self.index+1)%self.cap

	# def random_subset(n):
	# 	x= np.zeros(n)
	# 	y= np.zeros(n)
	# 	for i in range(n):
	# 		x[i] = np.random.randint(0,n)
	# 		ind= x[i]
	# 		y[i] = self.array[ind](1) + np.max(model.predict([ (frame), np.ones(NUM_ACTIONS,)]))

#funcs for image preprocessing-
#LTR: checkout ki jo frame return hota hai uska image kya hai and what part does it cover!!!

#asssuming its an np array
def halve_resolution(img):
	#jo bhi le uske pehle do axis pe alternate leke aadha kardtea hau
	return img[::2,::2]

def to_grayscale(img):
	return np.mean(img, axis=2).astype(np.uint8)
	#mean about axis2 means the axis that contains the 2 colours- R,G and B
	#i didnot know ki grayscale values directly mean leke aajati hain
	#THINK: potential miss of data- larger set to smaller set mapping, ill make an image jiska graycale ek cheez hoga, but uss image mei kaafi colours honnge
	#.astype yeh karta hai ki - online - Copy of the array, cast to a specified type.
	#so we just reduced the space used
def cutter(img):
	return img[17:97,4:76]

def binarise(img):
	for i in range(img.shape[1]):
		for j in range(img.shape[0]):
			temp = img[j][i] 
			# print (temp)
			if temp > 0 :
				img[j][i]=255
	return img


def preprocess(img):
	#input : 210 x 160 x 3
	#output : (105,80, )
	return to_grayscale((halve_resolution(img)))

# def transform_reward(reward):
# 	return np.sign(reward)
def huber_loss(a, b, in_keras=True):
	error = a - b
	quadratic_term = error*error / 2
	linear_term = abs(error) - 1/2
	use_linear_term = (abs(error) > 1.0)
	if in_keras:
		use_linear_term = backend.cast(use_linear_term, 'float32')
	return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


# buff = cyclic_queue(BUFFER_SIZE)
# #(layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

# def bakwaas_sequential():
	#yeh sab sequential model banane ki attempt thi- SAD
	# model= Sequential()

	# #add all layers ab
	# #(layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
	# #i took a screenshot of the game to actually see the dimensions and i was only able to reduce it to 208 x 180 =approx 210 x 160
	# #but i cant take a middle square out of this which contains all the useful info


	# model.add(Conv2D(16,(8,8) , strides=(4, 4),input_shape=(105, 80, 4), activation="relu"))
	# model.add(Conv2D(32, (4,4), strides=(2,2), activation="relu"))
	# #flatten?
	# # model.Flatten()
	# model.add(Dense(256,activation="relu"))
	# model.add(Dense(NUM_ACTIONS))
	# #THINK: here the activation was linear, AISA KYON
	# #filtering/masking layer
	# # model.add(Multiply())???????????????????????//variable mask//????????????????????????????????????????????????
	# model.compile(loss="mse", optimizer= "RMSprop", metrics=['accuracy'])
	# #LTR: metric kya dalun yahan bc?
	# return 

#now try creating the model using functional vali cheezen
#functional vale api mei the change is that model def is different, explicitly we define a first input layer
#i dont understand pehle hi kyo nhi padh liya yeh maine - faltu confuse hua
# baki fitting/training vagerah i beleive pehle jaise hi hoga
# model = Model()
#define layers


# frames_input = Input(FRAME_INPUT_SHAPE)
# mask_input = Input((NUM_ACTIONS, ))
#THINK: aise khali comma ke sath kyo define karna padta hai
#i just defined two inputs
#further i can decide ki further kon kese connect hoga

# #THINK: jo neeche yeh socha hua hai, this is the real thing? ya yeh sochna chill hai
# convolution1 = Conv2D(filters=16, kernel_size=(8,8),strides=(4,4), activation="relu", data_format='channels_first')(frames_input)
# convolution2 = Conv2D(filters=32, kernel_size=(4,4),strides= (2,2), activation="relu", data_format='channels_first')(convolution1)
# #i forgot this line!!!!
# flattened = Flatten()(convolution2)
# normaliser = Lambda(lambda x: (x*1.)/255.0 )(flattened)
# #custom functions
# hidden1 = Dense(256, activation="relu")(normaliser)
# hidden2 = Dense(NUM_ACTIONS)(hidden1)
# masked_output = multiply([mask_input,hidden2])

# model= Model(inputs= [frames_input, mask_input], outputs= masked_output)

#custom hyperparameters
# optim=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
# model.compile(loss= "mse", optimizer= "RMSprop")

# if(len(sys.argv)>1 and sys.argv[1]=="cont"):
model= load_model('training/light_v0_part2.h5',custom_objects={'huber_loss': huber_loss})

#main iteration
env = gym.make('Breakout-v0')
frame = env.reset() #initial frame
env.step(1)
env.step(0)
env.step(2)
frame= preprocess(frame)

state= np.array([frame, frame, frame, frame])
#faltu initialisation data point 

epsilon=.05 #greed metric

for j in range(ITERATIONS):
	env.render('human')

	if np.random.uniform(0,1) <= epsilon :
		action= env.action_space.sample()
		#produces a random action(uniform prob)
	else:
		#choose best action
		#feed the net with a full ones mask, and then choose the best
		# print("enter GREED")
		action= np.argmax(model.predict( [np.array([state]), np.array([np.ones(NUM_ACTIONS, )])]))
		# temp= model.predict( [np.array([state]), np.ones((1,NUM_ACTIONS))])
		# print("shape of prediction is ", temp.shape)
		# print("shape1 is ", np.array([state]).shape)
		# print("shape2 is ", np.array([np.ones(NUM_ACTIONS, )]).shape)
		# print ("action is ", action)

	#perform this
	#4 frames consecutively forms a state
	#THINK : i think thi simplementation is not correct, check out someone elses implementation on the net?
	frame_1, reward_1, termination, _ = env.step(action)
	

	
	frame_2, reward_2, termination, _ = env.step(action)
	
	
	frame_3, reward_3, termination, _ = env.step(action)
	
	
	frame_4, reward_4, termination, _ = env.step(action)
	
	
	# env.step(1)
	# env.step(1)
	if(termination):
		env.reset()
		# env.step(1)
		env.step(0)
		# env.step(1)
		# env.step()

	frame_1 = preprocess(frame_1)
	frame_2 = preprocess(frame_2)
	frame_3 = preprocess(frame_3)
	frame_4 = preprocess(frame_4)

	img = Image.fromarray(binarise(cutter(frame_1)))

	img.save('outfile1.jpg')
	img = Image.fromarray((cutter(frame_2)))
	

	img.save('outfile2.jpg')
	img = Image.fromarray((cutter(frame_3)))

	img.save('outfile3.jpg')
	img = Image.fromarray((cutter(frame_4)))

	img.save('outfile4.jpg')
	while true:
		pass


	input_state_array= np.zeros((MINIBATCH_SIZE, 4, 105, 80))
	state_new = np.array([frame_1, frame_2, frame_3, frame_4])

	# frame_new= preprocess(frame_new)
	#store in buffer as <s,a,r,s',term>
	# i need the termination value to know whether I have to set the reward as r + gamma*next or just r
	# buff.add((state, action, reward, state_new , termination))
	state= state_new	
	#THINK: the extra info being returned?
	l=0
	while l<30000:
		l=l+1

# 	if(j>BUFFER_SIZE):
# 		# print("enter TRAIN")
# 		#ek baar buffer fill ho jaye phir ghus raha hu ismei
# 		# x= np.zeros(MINIBATCH_SIZE, dtype= np.ndarray)
# 		# y= np.zeros(MINIBATCH_SIZE, dtype= np.ndarray)
# 		# for i in range(MINIBATCH_SIZE):
# 		# 	x[i] = np.random.randint(0,MINIBATCH_SIZE)
# 		# 	ind= x[i]
# 		# 	y[i] = (buff.array[ind])[1]
# 			# y[i] = buff.array[ind](1) + np.max(model.predict([ buff.array[ind](3), np.ones(NUM_ACTIONS,)]))
# 	#randomly select a minibatch ab, from the buffer

# 		rewards = np.zeros(MINIBATCH_SIZE)
# 		terminal = np.zeros(MINIBATCH_SIZE)
# 		one_hot_encoding = np.zeros((MINIBATCH_SIZE, NUM_ACTIONS))

# 		#retry writing this piece, train a batch
# 		# input_state_array= np.zeros()
# 		# all_qvalues = np.zeros(MINIBATCH_SIZE, dtype = np.ndarray)

# 		for i in range(MINIBATCH_SIZE):
# 			num= np.random.randint(MINIBATCH_SIZE)
# 			#(s,a,r,s',t)
# 			input_state_array[i]= buff.array[num][3]
# 			rewards[i] = buff.array[num][2]
# 			terminal[i] = buff.array[num][4]
# 			a= np.zeros(NUM_ACTIONS)
# 			a[buff.array[num][1]] = 1
# 			one_hot_encoding[i]=  a

# 		all_next_qvalues = model.predict( [ input_state_array, np.ones((MINIBATCH_SIZE, NUM_ACTIONS)) ])
		
# 		# print(all_next_qvalues.shape)
# 		# while true:
# 		# 	pass
# 		# #stop
# 		optimum_next_qvalues= np.max(all_next_qvalues, axis=1)
# 		for l in range(MINIBATCH_SIZE):
# 			if terminal[l] :
# 				optimum_next_qvalues[l]=0
# 				#basucally if its a terminal state 
# 		target = rewards + GAMMA*optimum_next_qvalues
# 		model.fit([input_state_array,one_hot_encoding ], one_hot_encoding * target[:, None] , epochs=1, batch_size= len(input_state_array), verbose= 0)

# model.save('light_v0_full.h5')