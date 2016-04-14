# Drop in replacement for TSNE
# Using BHSNE using the same interface contract as the original TSNE python implementation
# (PS: The original TSNE impl is with PCA dim. reduction, and this one is not)
#
# This code is based on the original Python wrapper of the bhtsne project
#
# Jos van Roosmalen - 11-Apr-2016

import numpy as Math
from struct import pack,unpack,calcsize
from os.path import join as path_join
import sys
import gc
from subprocess import check_call

def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
  postfix = "BHTSNEDropIn"
  # First we create the datafile
  f = createDataFile(".",postfix,len(X),no_dims,initial_dims,perplexity)
  appendData(f,X)
  closeDataFile(f)
  # now we call the C compiled executable
  callBHTSNE(postfix)
  # now we convert the result file back into a numpy array and return it
  return processResultFile(postfix)

# Support to build the file
def createDataFile(path,postfix,count,no_dims,initial_dims,perplexity):
  theta = 0.5
  data_file = open(path+'/data'+str(postfix)+'.dat', 'wb')
  data_file.write(pack('iiddi', count, initial_dims, theta, perplexity, no_dims))
  return data_file

def appendData(data_file,X):
  for sample in X:
      data_file.write(pack('{}d'.format(len(sample)), *sample))

def closeDataFile(data_file):
  data_file.close()

def callBHTSNE(arg):
  check_call(["./bh_tsne",arg])    

def processResultFile(postfix): 
  with open('result'+postfix+'.dat', 'rb') as output_file:
    # The first two integers are just the number of samples and the
    #   dimensionality
    result_samples, result_dims = _read_unpack('ii', output_file)
    # Collect the results, but they may be out of order
    results = [_read_unpack('{}d'.format(result_dims), output_file)
        for _ in range(0,result_samples)]
    # Now collect the landmark data so that we can return the data in
    #   the order it arrived
    results = [(_read_unpack('i', output_file), e) for e in results]
    # Put the results in order and yield it
    results.sort()
  # strip of the landmark
  r = [ x[1] for x in results ]
  r = Math.asarray(r)
  return r;

def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


 
