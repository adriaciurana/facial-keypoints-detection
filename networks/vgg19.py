import sys
import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import os, sys
from base import Base
from keras.callbacks import ModelCheckpoint

from keras.utils import plot_model
from keras import *
from keras.models import *
from keras.layers import *
from keras.losses import *
from keras.backend import tf as ktf

sys.path.append("..")
from misc.config import Config

class Network(Base):
	name = 'VGG19'
	def network(self):
		input_layer = Input(shape=(96, 96, 1), dtype='float32')
		resize_layer = Concatenate(axis=-1)([input_layer, input_layer, input_layer])
		resize_layer = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(resize_layer)
		base_model = VGG19(weights='imagenet', input_tensor=resize_layer, include_top=True)
		fc2 = base_model.get_layer('fc2').output
		
		b1 = Dense(int(Config.UNROLL_JOINTS), name='output_1', activation='linear')(fc2)

		b2 = Dense(4096, activation='sigmoid')(fc2)
		b2 = Dense(int(Config.UNROLL_JOINTS * 0.5), name='output_2', activation='sigmoid')(b2)

		output = Concatenate(axis=-1, name='global_output')([b1, b2])
		return Model(inputs= input_layer, outputs= output)

	def compile(self):
		model = self.net
		def root_mean_squared_error(y_true, y_pred):
			diff = y_pred - y_true
			return K.sqrt(K.mean(K.square(diff), axis=-1))

		def root_mean_squared_error_with_unknowns(y_true, y_pred, known):
			diff = y_pred - y_true
			# Only count the known values
			diff1 = diff[:, 0:Config.UNROLL_JOINTS:2]*known
			diff2 = diff[:, 1:Config.UNROLL_JOINTS:2]*known
			diff = K.concatenate([diff1, diff2], axis=-1)
			loss = K.sqrt(K.mean(K.square(diff), axis=-1))
			return loss

		def mergedLoss(y_true, y_pred):
			loss1 = binary_crossentropy(y_true[:, Config.UNROLL_JOINTS:], y_pred[:, Config.UNROLL_JOINTS:])
			loss2 = root_mean_squared_error_with_unknowns(y_true[:, :Config.UNROLL_JOINTS], y_pred[:, :Config.UNROLL_JOINTS], y_true[:, Config.UNROLL_JOINTS:])
			return loss1 + loss2

		def root_mean_squared_error_with_unknowns_metric(y_true, y_pred):
			return root_mean_squared_error_with_unknowns(y_true[:, :Config.UNROLL_JOINTS], y_pred[:, :Config.UNROLL_JOINTS], y_true[:, Config.UNROLL_JOINTS:])
		
		def root_mean_squared_error_metric(y_true, y_pred):
			return root_mean_squared_error(y_true[:, :Config.UNROLL_JOINTS], y_pred[:, :Config.UNROLL_JOINTS])
		
		def binary_crossentropy_metric(y_true, y_pred):
			return binary_crossentropy(y_true[:, Config.UNROLL_JOINTS:], y_pred[:, Config.UNROLL_JOINTS:])
		

		self.net.compile(loss= mergedLoss, optimizer=Config.OPTIMIZER, metrics=[root_mean_squared_error_with_unknowns_metric, binary_crossentropy_metric, root_mean_squared_error_metric])

	def fit(self, X, Y, params):
		model_checkpoint = ModelCheckpoint(self.save_file, monitor='val_loss', save_best_only=True)
		params['callbacks'] = [model_checkpoint] if 'callbacks' not in params else params['callbacks'] + [model_checkpoint]
		self.net.fit(x= X, y= Y, **params)
