from __future__ import print_function, division
from warnings import warn, filterwarnings

import random
import sys
import numpy as np
import time
import json

from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

from nilmtk import DataSet
import metrics
from gen import opends, gen_batch
from model import create_model

allowed_key_names = ['fridge','microwave','dish_washer','kettle','washing_machine']

def normalize(data, mmax, mean, std):
	return data / mmax

def denormalize(data, mmax, mean, std):
	return data * mmax

def experiment(key_name, start_e, end_e):
	'''Trains a network and disaggregates the testset
	Displays the metrics for the disaggregated part

	Parameters
	----------
	key_name : The string key of the appliance
	start_e : The starting number of epochs for Training
	end_e: The ending number of epochs for Training
	'''

	# =======  Open configuration file
	if (key_name not in allowed_key_names):
		print("    Device {} not available".format(key_name))
		print("    Available device names: {}", allowed_key_names)
	conf_filename = "appconf/{}.json".format(key_name)
	with open(conf_filename) as data_file:
		conf = json.load(data_file)

	input_window = conf['lookback']
	threshold = conf['on_threshold']
	mamax = 5000
	memax = conf['memax']
	mean = conf['mean']
	std = conf['std']
	train_buildings = conf['train_buildings']
	test_building = conf['test_building']
	on_threshold = conf['on_threshold']
	meter_key = conf['nilmtk_key']
	save_path = conf['save_path']

	# ======= Training phase
	print("Training for device: {}".format(key_name))
	print("    train_buildings: {}".format(train_buildings))

	# Open train sets
	X_train = np.load("dataset/trainsets/X-{}.npy".format(key_name))
	X_train = normalize(X_train, mamax, mean, std)
	y_train = np.load("dataset/trainsets/Y-{}.npy".format(key_name))
	y_train = normalize(y_train, memax, mean, std)
	model = create_model(input_window)

	# Train model and save checkpoints
	if start_e > 0:
		model = load_model(save_path+"CHECKPOINT-{}-{}epochs.hdf5".format(key_name, start_e))

	if end_e > start_e:
		filepath = save_path+"CHECKPOINT-"+key_name+"-{epoch:01d}epochs.hdf5"
		checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
		history = model.fit(X_train, y_train, batch_size=128, epochs=(end_e - start_e), shuffle=True, initial_epoch=start_e, callbacks=[checkpoint])
		losses = history.history['loss']

		model.save("{}CHECKPOINT-{}-{}epochs.hdf5".format(save_path, key_name, end_e),model)

		# Save training loss per epoch
		try:
			a = np.loadtxt("{}losses.csv".format(save_path))
			losses = np.append(a,losses)
		except:
			pass
		np.savetxt("{}losses.csv".format(save_path), losses, delimiter=",")

	# ======= Disaggregation phase
	mains, meter = opends(test_building, key_name)
	X_test = normalize(mains, mamax, mean, std)
	y_test = meter

	# Predict data
	X_batch, Y_batch = gen_batch(X_test, y_test, len(X_test)-input_window, 0, input_window)
	pred = model.predict(X_batch)
	pred = denormalize(pred, memax, mean, std)
	pred[pred<0] = 0
	pred = np.transpose(pred)[0]
	# Save results
	np.save("{}pred-{}-epochs{}".format(save_path, key_name, end_e), pred)

	rpaf = metrics.recall_precision_accuracy_f1(pred, Y_batch, threshold)
	rete = metrics.relative_error_total_energy(pred, Y_batch)
	mae = metrics.mean_absolute_error(pred, Y_batch)

	print("============ Recall: {}".format(rpaf[0]))
	print("============ Precision: {}".format(rpaf[1]))
	print("============ Accuracy: {}".format(rpaf[2]))
	print("============ F1 Score: {}".format(rpaf[3]))

	print("============ Relative error in total energy: {}".format(rete))
	print("============ Mean absolute error(in Watts): {}".format(mae))

	res_out = open("{}results-pred-{}-{}epochs".format(save_path, key_name, end_e), 'w')
	for r in rpaf:
		res_out.write(str(r))
		res_out.write(',')
	res_out.write(str(rete))
	res_out.write(',')
	res_out.write(str(mae))
	res_out.close()


if __name__ == "__main__":
	key_name = sys.argv[1]
	if (key_name == ""):
		print("    Usage: synth-test.py <devicename>")
		print("    Available device names: {}", allowed_key_names)
		exit()
	experiment(key_name, 0, 3)
