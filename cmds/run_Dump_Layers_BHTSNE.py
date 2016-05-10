# Copyright 2014    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#
import pickle
import gzip
import os
import sys
import time
import math
import bloscpack as bp
import glob

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils.BHTSNEDropInReplacementTSNE import createDataFile,appendData,closeDataFile
from models.dnn import DNN
from models.cnn import CNN

from io_func import smart_open
from io_func.model_io import _file2nnet, log
from utils.utils import parse_arguments, string2bool
from utils.network_config import NetworkConfig

def dumpInput(outputPath,perplexity,data_spec):
    x = sorted(glob.glob(data_spec));
    tmp = bp.unpack_ndarray_file(x[0])
    dims = len(tmp[0])
    count = len(tmp)    
    for i in range (1,len(x)):
        tmp = bp.unpack_ndarray_file(x[i])
        if len(tmp[0]) != dims:    
            raise ValueError("Dimension "+str(len(tmp[0]))+" not same as first file "+str(dims))
        count =  count + len(tmp)
    files = []
    for p in perplexity:
        files.append(createDataFile(outputPath,'OriginalData',count,2,dims,p))
    c = 0
    for inputFile in x:
        tmp = bp.unpack_ndarray_file(inputFile)
        for f in files:
            appendData(f,tmp)
        c = c + len(tmp)
    for data_file in files:
        closeDataFile(data_file)
    if c != count:
        raise ValueError("Error output count ("+str(c)+") is not equal to calculated output count ("+str(count)+")")
    return (count,dims)

if __name__ == '__main__':
    
    perplexity = [20]
    
    import sys

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['data', 'nnet_param', 'nnet_cfg', 'output_path',  'batch_size']
    for arg in required_arguments:
        if (arg in arguments) == False:
            print("Error: the argument %s has to be specified" % (arg)); exit(1)

    # mandatory arguments
    data_spec = arguments['data']
    nnet_param = arguments['nnet_param']
    nnet_cfg = arguments['nnet_cfg']
    output_path = arguments['output_path']
    batch_size = float(arguments['batch_size'])
   
    log("Dump original data") 
    countItems,initialDim = dumpInput(output_path,perplexity,data_spec)

    log("Original data dumped. Items="+str(countItems)+" initialDim="+str(initialDim))
    
    # load network configuration and set up the model
    log('> ... setting up the model and loading parameters')
    cfg = pickle.load(smart_open(nnet_cfg,'rb'))
    layerNr = cfg.totalNumerOfLayers() 
    log('Total number of layers '+str(layerNr))
    for i in range(0,layerNr):
        count = 0        
        log("Going to output layer="+str(i))
        files = []
        layer_index = i

        numpy_rng = numpy.random.RandomState(89677)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        cfg = pickle.load(smart_open(nnet_cfg,'rb'))
        cfg.init_activation()
        model = None
        if cfg.model_type == 'DNN':
            model = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
        elif cfg.model_type == 'CNN':
            model = CNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg, testing = True)

        # load model parameters
        _file2nnet(model.layers, path = nnet_param)

        # initialize data reading
        cfg.init_data_reading_test(data_spec)

        model.dumpLayerSize()

        initialDim = model.getNeuronsForLayer(i)
        for p in perplexity:
            files.append(createDataFile(output_path,'Layer'+str(layer_index),countItems,2,initialDim,p))

        # get the function for feature extraction
        log('> ... getting the feat-extraction function')
        extract_func = model.build_extract_feat_function(layer_index)

        output_mats = []    # store the features for all the data in memory. TODO: output the features in a streaming mode
        log('> ... generating features from the specified layer')
        while (not cfg.test_sets.is_finish()):  # loop over the data
            cfg.test_sets.load_next_partition(cfg.test_xy)
            batch_num = int(math.ceil(1.0 * cfg.test_sets.cur_frame_num / batch_size))

            for batch_index in range(batch_num):  # loop over mini-batches
                    start_index = int(batch_index * batch_size)
                    end_index = int(min((batch_index+1) * batch_size, cfg.test_sets.cur_frame_num))  # the residue may be smaller than a mini-batch
                    output = extract_func(cfg.test_x.get_value()[start_index:end_index])
                    count = count + len(output)
                    for data_file in files: 
                        appendData(data_file,output)

        log("Outputed "+str(count)+" expected "+str(countItems))
        if  count != countItems:
            raise ValueError("Count does not match")
        log("Closing the files")
        for data_file in files:
            closeDataFile(data_file)
    log("All layers dumped")
