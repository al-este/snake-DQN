import game
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Activation, Flatten, Add
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing import image
from keras.optimizers import Adam
from keras import backend as K

from random import choices, choice, randint

from math import log
import matplotlib.pyplot as plt
from time import sleep

N_SET = 0

DELTA = 0.9
eps = 0.0

def create_model():
	model = Sequential()
	model.add(Conv2D(32, (5, 5), padding='same',
					 input_shape=(20, 20, 4)))
	model.add(Activation('relu'))
	#model.add(MaxPooling2D())

	model.add(Conv2D(64, (3, 3), padding='valid'))
	model.add(Activation('relu'))

	model.add(Conv2D(128, (3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())

	model.add(Conv2D(256, (3, 3), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())

	model.add(Conv2D(512, (3, 3), padding='valid'))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(4))
	model.add(Activation('linear'))

	model.summary()

	return model

def create_model1():
	model = Sequential()
	
	model.add(Dense(256, input_dim=29))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(4))
	model.add(Activation('linear'))

	model.summary()

	return model

def create_model1():
	X_input = Input((20, 60, 1))
	X = X_input

	X = Conv2D(32, (3, 3), input_shape=(20, 60, 1), activation="relu", kernel_initializer='he_uniform', padding='same')(X)
	X = MaxPooling2D()(X)

	X = Conv2D(64, (3, 7), activation="relu", kernel_initializer='he_uniform', padding='valid')(X)
	X = MaxPooling2D()(X)

	X = Conv2D(128, (3, 7), activation="relu", kernel_initializer='he_uniform', padding='valid')(X)
	X = MaxPooling2D()(X)

	X = Conv2D(256, (3, 7), activation="relu", kernel_initializer='he_uniform', padding='same')(X)
	X = MaxPooling2D((1,2))(X)
	X = Flatten()(X)

	state_value = Dense(256, kernel_initializer='he_uniform', activation="relu")(X)
	state_value = Dense(1, kernel_initializer='he_uniform')(state_value)
	state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(4,))(state_value)

	action_advantage = Dense(256, kernel_initializer='he_uniform', activation="relu")(X)
	action_advantage = Dense(4, kernel_initializer='he_uniform')(action_advantage)
	action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(4,))(action_advantage)

	X = Add()([state_value, action_advantage])

	model = Model(inputs = X_input, outputs = X)

	model.summary()

	return model

def copy_model(source, target):
	target.set_weights(source.get_weights())

def matrix_to_array(matrix):
	return np.expand_dims(image.img_to_array(matrix),axis=0)

def get_movement(predict):
	if predict == 0:
		return KEY_RIGHT
	elif predict == 1:
		return KEY_LEFT
	elif predict == 2:
		return KEY_UP
	else:
		return KEY_DOWN

def save_model(s_model):
	model_json = s_model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	s_model.save_weights("model.h5")
	print("Saved model to disk")

def new_game_set(size):
	gset = list()
	for i in range(size):
		gset.append(game.game(False, 20, 20))
	return gset

def train(g_set):
	state_set = list()
	a_set = list()
	pre_set = list()
	post_set = list()
	r_set = list()
	done_set = list()
	def create_set(game, epsilon):
		pre = model.predict(matrix_to_array(game.get_move_matrix()))
		if np.random.random() < epsilon:
			a = randint(0, 3)
		else:
			a = np.argmax(pre)

		pre_set.append(pre[0])
		a_set.append(a)
		state = game.get_move_matrix()
		state_set.append(image.img_to_array(state))

		r, done = game.step(get_movement(a))

		r_set.append(r)

		if done == 'playing':
			done_set.append(1)
		else:
			done_set.append(0)

		post_set.append(game.get_move_matrix())

	scores=list()
	i_set=list()
	i=0
	for g in g_set:
		prev_score = 0
		loses = 0
		eps = 0.0
		while True:
			if g.get_state() == 'playing':
				create_set(g, eps)
				i_set.append(i)
				if g.get_score()!=prev_score:
					prev_score=g.get_score()
					eps = 0.0
					loses = 0
				else:
					loses+=1
					if loses>200:
						eps+=0.001
			else:
				break
		i+=1
		scores.append(g.get_score())
		g.print_matrix()
		print(i,'->' ,g.get_score(), '  EPS: ', eps)

	max_score = max(scores)
	print(scores)
	print("{} -> {}".format(max_score, np.argmax(scores)+1))
	max_score_history.append(max_score)

	predicts = freezed_model.predict(np.array(post_set))

	r_set = np.array(r_set)*((np.array(scores)[np.array(i_set)]+1.0)/(max_score+1.0))

	targets = r_set + DELTA*np.max(np.array(predicts), axis=1)#*np.array(done_set)

	pre_set=np.array(pre_set)

	pre_set[range(len(pre_set)), a_set] = targets

	print(np.mean(scores))
	score.append(np.mean(scores))

	res = model.fit(np.array(state_set), np.array(pre_set), epochs = 10, batch_size = 100, verbose=1)
	history.append(res.history['loss'][0])

def load_model(model):
	try:
		model.load_weights("model.h5")
		copy_model(model, freezed_model)
	except OSError:
		print("\n\n No model found \n\n")
		sleep(1)
	else:
		print("Loaded model from disk")

	return model

model = create_model()
freezed_model = create_model()

opt = Adam(lr=0.001)

model.compile(loss='mean_squared_error', optimizer=opt)

#Carga el modelo anterior para seguir con el entrenamiento---> 
model = load_model(model)

copy_model(model, freezed_model)

history = list()
score = list()
max_score_history=list()

it = 0
try:
	while True:
		gset = new_game_set(100)
		train(gset)
		eps*=0.9
		it += 1
		if it >= 10:
			it = 0
			copy_model(model, freezed_model)
			print('model -> freezed_model')
			save_model(model)

		print("Epoch: {}, EPS: {:.5f}".format(len(history), eps))

except KeyboardInterrupt:
	fig, ax1 = plt.subplots()

	color = 'tab:blue'
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss', color=color)
	ax1.plot(history, color=color, label='Loss')
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:red'
	ax2.set_ylabel('Score', color=color)  # we already handled the x-label with ax1
	ax2.plot(score, color=color, label='Mean score')
	ax2.plot(max_score_history, color="tab:green", label='Max score')
	ax2.tick_params(axis='y', labelcolor=color)
	ax1.legend(loc='upper left')
	ax2.legend(loc='upper right')
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()