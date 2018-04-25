import sys, os
import numpy as np
from networks.vgg19 import Network
from misc.config import Config
from misc.misc import Misc
import cv2

# DIR_PATH
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

print 'Loading TEST data'
X = Misc.prepare_data_test_from_file(Config.PATH_TEST_CSV)
Xs = (X - 124.84467441020496) / 59.3492454319411


# GPU
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

print 'Loading Network'
network = Network(DIR_PATH)
network.load()
outlist = network.predict(Xs)

def drawResult(imgIn, result):
	img = np.zeros(shape=(imgIn.shape[0], imgIn.shape[1], 3), dtype='uint8')
	img[:,:,0] = np.reshape(imgIn, [imgIn.shape[0], imgIn.shape[1]])
	img[:,:,1] = img[:,:,0]
	img[:,:,2] = img[:,:,0]
	
	points = result[:Config.UNROLL_JOINTS]
	known = result[Config.UNROLL_JOINTS:]

	for i in xrange(0, len(points), 2):
		k = i // 2
		if known[k] > Config.THRES_MIN:
			x, y = points[i], points[i+1]
			cv2.circle(img, (x, y), 2, (0, 0, 255))
	return img

for i in xrange(outlist.shape[0]):
	outitem = outlist[i]
	cv2.imshow("image", drawResult(X[i], outitem))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
