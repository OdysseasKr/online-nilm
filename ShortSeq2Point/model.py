from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout, Flatten
from keras.utils import plot_model

def create_model(input_window):
	'''Creates and returns the ShortSeq2Point Network
	Based on: https://arxiv.org/pdf/1612.09106v3.pdf
	'''
	model = Sequential()

	# 1D Conv
	model.add(Conv1D(30, 10, activation='relu', input_shape=(input_window,1), padding="same", strides=1))
	model.add(Dropout(0.5))
	model.add(Conv1D(30, 8, activation='relu', padding="same", strides=1))
	model.add(Dropout(0.5))
	model.add(Conv1D(40, 6, activation='relu', padding="same", strides=1))
	model.add(Dropout(0.5))
	model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
	model.add(Dropout(0.5))
	# Fully Connected Layers
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mse', optimizer='adam')
	plot_model(model, to_file='model.png', show_shapes=True)

	return model
