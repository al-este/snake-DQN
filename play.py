import game
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN

import numpy as np

from time import sleep

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import model_from_json

from keras.applications.vgg16 import VGG16
from matplotlib import pyplot

def matrix_to_array(matrix):
	return np.expand_dims(image.img_to_array(matrix),axis=0)

def get_movement(predict):
	i = np.argmax(predict)
	if i == 0:
		return KEY_RIGHT
	elif i == 1:
		return KEY_LEFT
	elif i == 2:
		return KEY_UP
	else:
		return KEY_DOWN

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#loaded_model.summary()

g = game.game(False, 20, 20)
g.print_matrix()

state = 'playing'
while state == 'playing':
	predict = loaded_model.predict(matrix_to_array(g.get_move_matrix()))[0]
	state = g.get_matrix()
	a = get_movement(predict)
	g.print_matrix()
	print("Reward "+str(g.step(a)[0]))
	print(a)
	print(predict)
	print(g.get_score())
	#print(g.get_move_matrix())
	#sleep(0.05)
	state = g.get_state()
