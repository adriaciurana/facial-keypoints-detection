import os, sys
from keras.utils import plot_model
import abc
from keras import *
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras.backend import tf as ktf
class Base(object):
	name = 'Base'
	def __init__(self, dire):
		print 'Running the '+self.name+' network'
		self.dir = dire
		self.save_file = os.path.dirname(os.path.realpath(__file__))+"/"+self.name+'_weights.h5'
		self.create()

	def create(self):
		self.net = self.network()

	def save(self, filex = None):
		if filex == None:
			filex = self.save_file
		# save weights
		self.net.save_weights(filex)
		print("Save done!")

	def load(self, filex = None):
		if filex == None:
			filex = self.save_file
		# load weights into new model
		self.net.load_weights(filex)
		print("Loaded done!")

	def load_if_exist(self, filex = None):
		if filex == None:
			filex = self.save_file
		if(os.path.isfile(filex)):
			self.load(filex)
		
	def plot(self):
		plot_model(self.net, to_file=self.dir+'/'+self.name+'.png', show_shapes=True)

	def summary(self):
		print self.net.summary()

	def predict(self, X):
		return self.net.predict(X)

	@abc.abstractmethod
	def network(self):
		return

	@abc.abstractmethod
	def compile(self):
		return

	@abc.abstractmethod
	def fit(self, X, Y, params):
		return
