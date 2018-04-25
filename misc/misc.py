import numpy as np
import pandas as pd
import cv2
from config import *
from scipy.stats import multivariate_normal
import itertools
import tensorflow as tf
class Misc:
	@staticmethod
	def get_image_rgb(text):
		img = np.zeros(shape= (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype= np.uint8)
		values = [int(i) for i in text.split(' ')]
		for i in xrange(Config.IMAGE_SIZE[0]):
			for j in xrange(Config.IMAGE_SIZE[1]):
				aux = values[j + i*Config.IMAGE_SIZE[1]]
				img[i, j, :] = aux, aux, aux
		return img

	@staticmethod
	def get_image_gray(text):
		img = np.zeros(shape= (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype= np.uint8)
		values = [int(i) for i in text.split(' ')]
		for i in xrange(Config.IMAGE_SIZE[0]):
			for j in xrange(Config.IMAGE_SIZE[1]):
				aux = values[j + i*Config.IMAGE_SIZE[1]]
				img[i, j] = aux
		return img

	@staticmethod
	def visualize(row):
		img = Misc.get_image_rgb(row)
		cv2.imshow("image", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	@staticmethod
	def visitor_row(file, visitor):
		data=pd.read_csv(file)
		for i in xrange(len(data)):
			visitor(data.iloc[i])
			
	@staticmethod
	def prepare_data_from_file(file):
		X = []
		Y = []

		def visitor(row):
			X.append(Misc.get_image_gray(row.Image))
			Y.append(row.iloc[:-1])

		Misc.visitor_row(file, visitor)
		X = np.reshape(np.array(X), [len(X), Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 1])
		Y = np.nan_to_num(np.array(Y, dtype='float32'))


		# Normalization
		X_mean = np.mean(X[:])
		X_std = np.std(X[:])

		X_norm = (X - X_mean)/X_std
		return X_norm, Y, {'mean': X_mean, 'std': X_std}

	@staticmethod
	def prepare_data_from_file_with_unknowns(file):
		X = []
		Y = []

		def prepare_unknowns(row):
			row = np.array(row, dtype='float32')
			num = len(row) / 2
			# out = x + y + unknown
			arr = np.ones(shape=(3*num))
			arr[:len(row)] = np.nan_to_num(row)
			idx = np.argwhere(np.isnan(row))
			idx = np.uint8(idx / 2)
			arr[len(row)+idx] = 0
			return arr


		def visitor(row):
			X.append(Misc.get_image_gray(row.Image))
			Y.append(prepare_unknowns(row.iloc[:-1]))

		Misc.visitor_row(file, visitor)
		X = np.reshape(np.array(X), [len(X), Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 1])
		Y = np.array(Y, dtype='float32')


		# Normalization
		X_mean = np.mean(X[:])
		X_std = np.std(X[:])

		X_norm = (X - X_mean)/X_std
		return X_norm, Y, np.array([X_mean, X_std])

	"""@staticmethod
	def mrse_from_gaussian(y_true, y_pred):
		def get_argmax(arr):
			arr = K.reshape(arr, [-1, Config.IMAGE_SIZE[0]*Config.IMAGE_SIZE[1], Config.JOINTS])
			arr = K.argmax(arr, axis=-2)
			idx = tf.unravel_index(arr, (Config.IMAGE_SIZE[0]*Config.IMAGE_SIZE[1]))
			return K.concatenate(K.expand_dims(idx, axis=-1), axis=-1)

		y_true_points = get_argmax(y_true)
		y_pred_points = get_argmax(y_pred)

		diff = y_true_points - y_pred_points
		diff = diff[:]

		result = K.sqrt(K.sum(diff*diff)/float(diff.shape[0]))
		return K.varialize(result)"""

	@staticmethod
	def prepare_data_test_from_file(file):
		X = []
		def visitor(row):
			X.append(Misc.get_image_gray(row.Image))
		Misc.visitor_row(file, visitor)
		return np.reshape(np.array(X), [len(X), Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 1])




