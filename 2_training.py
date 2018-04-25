import sys, os
import numpy as np
from networks.vgg19 import Network
from misc.config import Config
from misc.misc import Misc

# DIR_PATH
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# load DATA
print 'Loading data'
data = np.load(Config.DATA_PREPARATION_FILE+".npz")
X = data['X']
Y = data['Y']
norm = data['norm']
del data

# GPU
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# network
network = Network(DIR_PATH)

# load
network.load_if_exist()

# plot network
network.plot()

# compile
network.compile()

# summary
network.summary()

# fit
network.fit(X, Y, Config.NETWORK_PARAMS)
