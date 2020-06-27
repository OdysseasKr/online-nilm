""" This script generates train sets from several building data
"""
from __future__ import print_function, division
from warnings import warn, filterwarnings

import numpy as np
import urllib
import os
import sys
import json
import shutil
from nilmtk import DataSet
from nilmtk.electric import align_two_meters

ukdale_windows = [
	("2014-9-1","2014-9-30"),
	("2013-9-1","2013-9-30"),
	("2013-3-1","2013-3-30"),
	("2013-4-7","2013-5-7"),
	("2014-9-1","2014-9-30"),
]

train_size = 200000
def create_trainset(meter, mains, train_size, window_size):
	'''Creates a time series from the raw UKDALE DataSet
	'''
	all_x_train = np.empty((train_size,window_size,1))
	all_y_train = np.empty((train_size,))
	low_index = 0

	gen = align_two_meters(meter, mains)
	for chunk in gen:
		if (chunk.shape[0]<3000):
			continue
		chunk.fillna(method='ffill', inplace=True)
		X_batch, Y_batch = gen_batch(chunk.iloc[:,1], chunk.iloc[:,0], chunk.shape[0]-window_size, 0, window_size)
		high_index = min(len(X_batch), train_size-low_index)
		all_x_train[low_index:high_index+low_index] = X_batch[:high_index]
		all_y_train[low_index:high_index+low_index] = Y_batch[:high_index]
		low_index = high_index+low_index
		if (low_index == train_size):
			break

	return all_x_train, all_y_train

def gen_batch(mainchunk, meterchunk, batch_size, index, window_size):
	'''Generates batches from dataset

	Parameters
	----------
	index : the index of the batch
	'''
	w = window_size
	offset = index*batch_size
	X_batch = np.array([ mainchunk[i+offset:i+offset+w]
						for i in range(batch_size) ])

	Y_batch = meterchunk[w-1+offset:w-1+offset+batch_size]
	X_batch = np.reshape(X_batch, (len(X_batch), w ,1))

	return X_batch, Y_batch

def opends(building, meter):
	'''Opens dataset of synthetic data from Neural NILM

	Parameters
	----------
	building : The integer id of the building
	meter : The string key of the meter

	Returns: np.arrays of data in the following order: main data, meter data
	'''

	path = "dataset/ground_truth_and_mains/"
	main_filename = "{}building_{}_mains.csv".format(path, building)
	meter_filename = "{}building_{}_{}.csv".format(path, building, meter)
	mains = np.genfromtxt(main_filename)
	meter = np.genfromtxt(meter_filename)
	mains = mains
	meter = meter
	up_limit = min(len(mains),len(meter))
	return mains[:up_limit], meter[:up_limit]

def download_dataset():
	print("Downloading dataset for the first time")
	os.makedirs("dataset")
	urllib.request.urlretrieve("http://jack-kelly.com/files/neuralnilm/NeuralNILM_data.zip", "dataset/ds.zip")
	import zipfile

	zip_ref = zipfile.ZipFile('dataset/ds.zip', 'r')
	zip_ref.extractall('dataset')
	zip_ref.close()
	os.remove("dataset/ds.zip")
	shutil.rmtree("dataset/disag_estimates", ignore_errors=True)
	os.makedirs("dataset/trainsets")
	print("Done downloading")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python gen.py ukdale_path")
		exit()

	conf_files =  os.listdir("appconf")
	ds = DataSet(sys.argv[1])
	for app in conf_files:
		filename = "appconf/{}".format(app)
		with open(filename) as data_file:
			conf = json.load(data_file)

		if not os.path.exists("dataset"):
			download_dataset()
		os.makedirs(conf['save_path'])

		# Create trainset for meter
		print(conf["nilmtk_key"])
		house_keys = conf['train_buildings']
		window_size = conf['lookback']
		all_x_train = np.empty((train_size*len(house_keys),window_size,1))
		all_y_train = np.empty((train_size*len(house_keys),))
		for i, building in enumerate(house_keys):
			ds.set_window(start=(ukdale_windows[building-1])[0], end=(ukdale_windows[building-1])[1])
			elec = ds.buildings[building].elec
			meter = elec[conf["nilmtk_key"]]
			mains = elec.mains()
			all_x, all_y = create_trainset(meter, mains, train_size, window_size)
			all_x = all_x
			all_y = all_y

			all_x_train[i*train_size:(i+1)*train_size] = all_x
			all_y_train[i*train_size:(i+1)*train_size] = all_y


		np.save('dataset/trainsets/X-{}'.format(conf['synth_key']),all_x_train)
		np.save('dataset/trainsets/Y-{}'.format(conf['synth_key']),all_y_train)
