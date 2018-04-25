import sys, os
import numpy as np
from misc.config import Config
from misc.misc import Misc

# Prepare data
X, Y, norm = Misc.prepare_data_from_file_with_unknowns(Config.PATH_TRAINING_CSV)
print X.shape, Y.shape
print 'Data preparation: DONE!'
np.savez(Config.DATA_PREPARATION_FILE, X=X, Y=Y, norm=norm)

print 'Done!'
print 'Summary:'
print 'Features to determine position: ' + str(Config.UNROLL_JOINTS) + ' (x and y), regresion'
print 'Features to detect: ' + str(Config.UNROLL_JOINTS*0.5) + ', classification'
print 'Total of Y "labels": '+ str(Config.UNROLL_JOINTS*1.5)
