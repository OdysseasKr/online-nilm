from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout
from keras.utils import plot_model

def create_model(input_window):
	'''Creates and returns the Neural Network
	'''
	model = Sequential()

	# 1D Conv
	model.add(Conv1D(16, 4, activation='relu', input_shape=(input_window,1), padding="same", strides=1))

	#Bi-directional GRUs
	model.add(Bidirectional(GRU(64, activation='relu', return_sequences=True), merge_mode='concat'))
	model.add(Dropout(0.5))
	model.add(Bidirectional(GRU(128, activation='relu', return_sequences=False), merge_mode='concat'))
	model.add(Dropout(0.5))

	# Fully Connected Layers
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mse', optimizer='adam')
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

	return model
