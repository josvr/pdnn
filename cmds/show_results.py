import glob
import numpy
import sys
import os
import pickle, gzip
import bloscpack as bp
from io_func.model_io import  log

pred_file = sys.argv[1]

if '.gz' in pred_file:
    pred_mat = pickle.load(gzip.open(pred_file, 'rb'))
else:
    pred_mat = pickle.load(open(pred_file, 'rb'))
 
l = sorted(glob.glob(sys.argv[2]))
if len(l) == 0:  
	log("ERROR in show_results. Test partitions is empty. Argument "+sys.argv[2])

test_labels = bp.unpack_ndarray_file(l[0]+".labels")
test_labels = test_labels.astype(numpy.int32)

for i in range (1,len(l)):
    lab = bp.unpack_ndarray_file(l[i]+".labels")
    lab = lab.astype(numpy.int32)
    test_labels = numpy.concatenate((test_labels,lab))

correct_number = 0.0
for i in range(0,pred_mat.shape[0]):
    p = pred_mat[i, :]
    p_sorted = (-p).argsort()
    if p_sorted[0] == test_labels[i]:
        correct_number += 1

# output the final error rate
error_rate = 100 * (1.0 - correct_number / pred_mat.shape[0])
print ('Error rate is ' + str(error_rate) + ' (%)')

