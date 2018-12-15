##THE CHANGE IS acc t0o blog2 of adrien
##bakchodis jo pehle thi- ---- Reward = -1/0/1 isko definitely change karna hai
##problem- wont be able to distinguish bw discounted rewards, 100 and 0.1 dono ko 1 treat karega
#print avg Q as a progress metric
#huber loss?
##fix the state def, use last 4 frames, 
#preprocessing mei you may add the taking max of two- the paper does it, adrien doesnt
#check the time of forward pass and training using a print time?
##smaintaing 2 models!!  predictor and target!!
##4 steps for each gradient descent- should leadd to better training?
##explicitly note the change in number of lives and give that out as a loss
##dont clip the rewards- removed that shit- this bot is just for atari
##i can read their code too- at https://sites.google.com/a/deepmind.com/dqn/
#mera code raat bhar chalanhi tha - probably a bug


import gym
import numpy as np 
import sys
# from keras.models import Sequential
import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import multiply
from keras.layers import Maximum
from keras.layers import Lambda
from keras import backend 
from matplotlib import pyplot as plt  

BUFFER_SIZE=10000
NUM_ACTIONS=4
FRAME_INPUT_SHAPE= (4,105,80) #(CHannels x Height x width)
ITERATIONS= 500000
DECREASE_RATE= 100000
MINIBATCH_SIZE=32
GAMMA=0.99
START_LIMIT = 500
C= 100
STEPS_BEFORE_GRADIENT_DESCENT=1
STAGE_ITERATION = 5000 # store a stage after these many iteration
EPSILON_INFO = 1000 #printing epsilon info after every these many iterations
#HARD: making it specific to breakout abhi ke liye
alpha= 0.01
# class transition():
# 	self.

class cyclic_queue():
	def __init__(self, capacity):
		self.cap = capacity
		self.index=0
		self.array = np.zeros(capacity, dtype=tuple)
		self.filled=0
		#now it expects an object there- i need a better space optimisation, cause i know ki kitne size ki state ayegi, action , vagerah


	def add(self,k):
		self.array[self.index]= k
		self.index = (self.index+1)%self.cap
		if(self.filled< BUFFER_SIZE):
			self.filled= self.filled+ 1

#funcs for image preprocessing-
#LTR: checkout ki jo frame return hota hai uska image kya hai and what part does it cover!!!

#asssuming its an np array

#adriens code for huber loss- because now im too tired
def huber_loss(a, b, in_keras=True):
	error = a - b
	quadratic_term = error*error / 2
	linear_term = abs(error) - 1/2
	use_linear_term = (abs(error) > 1.0)
	if in_keras:
		use_linear_term = backend.cast(use_linear_term, 'float32')
	return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


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

def preprocess(img):
	#input : 210 x 160 x 3
	#output : (105,80, )
	return to_grayscale(halve_resolution(img))

def transform_reward(reward):
	return np.sign(reward)

def copy_model(model):
	model.save("temp_model.h5")
	new_model= load_model("temp_model.h5", custom_objects={'huber_loss': huber_loss})
	return new_model


buff = cyclic_queue(BUFFER_SIZE)

frames_input = Input(FRAME_INPUT_SHAPE)
mask_input = Input((NUM_ACTIONS, ))
#THINK: aise khali comma ke sath kyo define karna padta hai


#THINK: jo neeche yeh socha hua hai, this is the real thing? ya yeh sochna chill hai
convolution1 = Conv2D(filters=16, kernel_size=(8,8),strides=(4,4), activation="relu", data_format='channels_first')(frames_input)
convolution2 = Conv2D(filters=32, kernel_size=(4,4),strides= (2,2), activation="relu", data_format='channels_first')(convolution1)
#i forgot this line!!!!
flattened = Flatten()(convolution2)
normaliser = Lambda(lambda x: (x*1.)/255.0 )(flattened)
#custom functions
hidden1 = Dense(256, activation="relu")(normaliser)
hidden2 = Dense(NUM_ACTIONS)(hidden1)
masked_output = multiply([mask_input,hidden2])

model= Model(inputs= [frames_input, mask_input], outputs= masked_output)

#custom hyperparameters
optim=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
model.compile(loss= huber_loss, optimizer= optim)

#fixing the buffer to incorporate frames
if(len(sys.argv)>1 and sys.argv[1]=="cont"):
	model= load_model('training/light_v0_part6.h5', custom_objects={'huber_loss': huber_loss})
	print("loaded existing model, continuing training")

target_model = copy_model(model)
#main iteration
env = gym.make('BreakoutDeterministic-v4')
initial_state_array= np.zeros((MINIBATCH_SIZE, 4, 105, 80))
next_state_array= np.zeros((MINIBATCH_SIZE, 4, 105, 80))


#faltu initialisation data point 
# frame = env.reset()
# frame= preprocess(frame)
# buff.add((frame,0,0,False))
# buff.add((frame,0,0,False))
# buff.add((frame,0,0,False))

env.reset()
previous_num_lives=5
epsilon=1 #greed metric
average_predicted_q=0

for j in range(ITERATIONS):
	# env.render()
	if(j%C==0):
		# print("Updating my TARGET")
		target_model= copy_model(model)


	if(j%10000==0):
		print("check RAM, buffer percent filled = ", (j*1./BUFFER_SIZE)*(100), "%")
		print("iterations ", j)

	if(j%EPSILON_INFO==0):
		print("exploration factor- ", epsilon)
		print("aeverage_predicted_q- ", average_predicted_q)
	if(j%STAGE_ITERATION==0):
		print("CHECKPOINT REACHED-----------Stage Complete: ",j/STAGE_ITERATION)
		model.save("training/light_v0_part"+str((int)(j/STAGE_ITERATION))+".h5")
	#update epsilon acc to num of iterations

	# state= 
	if(epsilon>0.1):
		epsilon = epsilon - 0.9/DECREASE_RATE

	if np.random.uniform(0,1) <= epsilon or j<10:
		action= env.action_space.sample()
		#produces a random action(uniform prob)
	else:
		
		#choose best action
		state= np.array([buff.array[buff.index-1][0],buff.array[buff.index-2][0],buff.array[buff.index-3][0],buff.array[buff.index-4][0]])
		action= np.argmax(model.predict( [np.array([state]), np.array([np.ones(NUM_ACTIONS, )])]))
		# temp= model.predict( [np.array([state]), np.ones((1,NUM_ACTIONS))])


	
	new_frame, reward, termination, live_dict = env.step(action)
	new_frame= preprocess(new_frame)
	# print(live_dict)
	new_lives= live_dict['ale.lives']

	if(previous_num_lives> new_lives):
		reward = reward - (0.4)**(5-new_lives)**2
		if(new_lives>5):
			print("ERROR________________________________________")
	previous_num_lives= new_lives
	if(termination):
		fresh_frame= env.reset()
		env.step(0) #to start a new game
		reward = -5
		# reward = -5
		

	buff.add((new_frame, action, reward, termination))


	if(j>START_LIMIT and j%STEPS_BEFORE_GRADIENT_DESCENT==0):
		# print("Enter train")
		rewards = np.zeros(MINIBATCH_SIZE)
		terminal = np.zeros(MINIBATCH_SIZE)
		one_hot_encoding = np.zeros((MINIBATCH_SIZE, NUM_ACTIONS))

		#retry writing this piece, train a batch
		# input_state_array= np.zeros()
		# all_qvalues = np.zeros(MINIBATCH_SIZE, dtype = np.ndarray)

		for i in range(MINIBATCH_SIZE):
			num= np.random.randint(4,buff.filled-2)
			#(f,a,r,t)
			#haan yahan copying mei time waste ho raha hai but its not the sowest step- chill
			initial_state_array[i]= np.array([buff.array[num][0], buff.array[num-1][0], buff.array[num-2][0], buff.array[num-3][0]])
			next_state_array[i]= np.array([buff.array[num+1][0], buff.array[num][0], buff.array[num-1][0], buff.array[num-2][0]])
			rewards[i] = buff.array[num][2]
			terminal[i] = buff.array[num][3]

			a= np.zeros(NUM_ACTIONS)
			a[buff.array[num][1]] = 1
			one_hot_encoding[i]=  a

		all_next_qvalues = target_model.predict( [ next_state_array, np.ones((MINIBATCH_SIZE, NUM_ACTIONS)) ])
		
		# print(all_next_qvalues.shape)
		# while true:
		# 	pass
		# #stop
		optimum_next_qvalues= np.max(all_next_qvalues, axis=1)
		average_predicted_q= (1-alpha)*average_predicted_q + (alpha)*(np.sum(optimum_next_qvalues) / MINIBATCH_SIZE)
 
		target = rewards + GAMMA*optimum_next_qvalues
		for l in range(MINIBATCH_SIZE):
			if terminal[l] :
				target[l]=rewards[l]
				#basucally if its a terminal state
		model.fit([initial_state_array,one_hot_encoding ], one_hot_encoding * target[:, None] , epochs=1, batch_size= len(initial_state_array), verbose=0)

model.save('training/light_v0_full.h5')