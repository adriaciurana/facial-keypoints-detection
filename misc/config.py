import pandas as pd
from keras.optimizers import *
class Config:
	# DATA CONFIG
	THRES_MIN = 0.3
	IMAGE_SIZE = 96, 96
	PATH_TRAINING_CSV = '/media/Datos/Deep Learning/Kaggle/limpios/facial-keypoints-detection/dataset/training.csv'
	DATA_PREPARATION_FILE = '/media/Datos/Deep Learning/Kaggle/limpios/facial-keypoints-detection/dataset/data'
	PATH_TEST_CSV = '/media/Datos/Deep Learning/Kaggle/limpios/facial-keypoints-detection/dataset/test.csv'

	# STRUCTURE NETWORK
	UNROLL_JOINTS = (len(pd.read_csv(PATH_TRAINING_CSV, nrows = 1).columns) - 1) # features to detect
	OPTIMIZER = Adam(lr=1e-5)
	NETWORK_PARAMS = {'epochs': 20, 'batch_size': 32, 'validation_split': 0.2, 'verbose': 1, 'shuffle':True}
