# Copyright 2013    Yajie Miao        Carnegie Mellon University
#           2015    Yun Wang          Carnegie Mellon University
#           2016    Jos van Roosmalen Open University The Netherlands

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

# Various functions to write models from nets to files, and to read models from
# files to nets
#
#
#
# 2016 change: net2file and file2net is now using high performance bloscpac over (slow) json

import shutil
import numpy as np
import os
import uuid
import bloscpack as bp
import sys
import pickle

from io import BytesIO
import json

import theano
import theano.tensor as T

from datetime import datetime

from io_func import smart_open

# print log to standard output
def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')
    sys.stderr.flush()

# convert an array to a string
def array_2_string(array):
    str_out = BytesIO()
    np.savetxt(str_out, array)
    return str_out.getvalue().decode("utf-8")

# convert a string to an array
#def string_2_array(string):
#    str_in = StringIO(string)
#    return np.loadtxt(str_in)

def string_2_array(string):
    str_in = BytesIO(string.encode("utf-8"))
    array_tmp = np.loadtxt(str_in)
    if len(array_tmp.shape) == 0:
        return np.array([array_tmp])
    return array_tmp

def _nnet2file(layers, set_layer_num = -1, path="dnn.tmp", start_layer = 0, input_factor = 0.0, factor=[]):
    if os.path.exists(path):
       shutil.rmtree(path)
    os.makedirs(path)
    blosc_args=bp.BloscArgs(clevel=9) 
    n_layers = len(layers)
    nnet_dict = {}
    if set_layer_num == -1:
       set_layer_num = n_layers
    for i in range(start_layer, set_layer_num):
       layer = layers[i]
       dict_a = 'W' + str(i)
       dropout_factor = 0.0
       if i == 0:
           dropout_factor = input_factor
       if i > 0 and len(factor) > 0:
           dropout_factor = factor[i-1]

       if layer.type == 'fc':
           n = str(uuid.uuid4())+".blp"
           tmpFileName = path + "/" + n;
           nnet_dict[dict_a] = n
           bp.pack_ndarray_file((1.0 - dropout_factor) * layer.W.get_value(), tmpFileName, chunk_size='100M', blosc_args=blosc_args)
       elif layer.type == 'conv':
           filter_shape = layer.filter_shape
           for next_X in range(filter_shape[0]):
               for this_X in range(filter_shape[1]):
                   n = str(uuid.uuid4())+".blp" 
                   tmpFileName = path + "/" + n;
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = n
                   bp.pack_ndarray_file((1.0-dropout_factor) * (layer.W.get_value())[next_X, this_X], tmpFileName, chunk_size='100M', blosc_args=blosc_args)

       n = str(uuid.uuid4())+".blp"
       tmpFileName = path + "/" + n;
       dict_a = 'b' + str(i)
       nnet_dict[dict_a] = n
       bp.pack_ndarray_file(layer.b.get_value(),tmpFileName, chunk_size='100M', blosc_args=blosc_args)

    with open(path + '/metadata.tmp', 'wb') as fp:
        pickle.dump(nnet_dict,fp,pickle.HIGHEST_PROTOCOL)


# save the config classes; since we are using pickle to serialize the whole class, it's better to set the
# data reading and learning rate interfaces to None.
def _cfg2file(cfg, filename='cfg.out'):
    s1 = cfg.lrate
    s2 = cfg.train_sets
    s3 = cfg.train_xy
    s4 = cfg.train_x
    s5 = cfg.train_y
    s6 = cfg.valid_sets
    s7 = cfg.valid_xy
    s8 = cfg.valid_x
    s9 = cfg.valid_y
    s10 = cfg.activation

    cfg.lrate = None
    cfg.train_sets = None
    cfg.train_xy = None
    cfg.train_x = None
    cfg.train_y = None
    cfg.valid_sets = None 
    cfg.valid_xy = None 
    cfg.valid_x = None 
    cfg.valid_y = None
    cfg.activation = None  # saving the rectifier function causes errors; thus we don't save the activation function
                           # the activation function is initialized from the activation text ("sigmoid") when the network
                           # configuration is loaded
    with open(filename, "wb") as output:
        pickle.dump(cfg, output, pickle.HIGHEST_PROTOCOL)
    
    cfg.lrate = s1
    cfg.train_sets = s2
    cfg.train_xy = s3
    cfg.train_x = s4
    cfg.train_y = s5
    cfg.valid_sets = s6
    cfg.valid_xy = s7
    cfg.valid_x = s8 
    cfg.valid_y = s9
    cfg.activation = s10 

def _file2nnet(layers, set_layer_num = -1, path="dnn.tmp",  factor=1.0):
    n_layers = len(layers)
    nnet_dict = {}
    if set_layer_num == -1:
        set_layer_num = n_layers
        log("file2nnet set_layer_num is -1 so set it to "+str(set_layer_num))
    
    with open(path + '/metadata.tmp', 'rb') as fp:
        nnet_dict = pickle.load(fp)
    for i in range(set_layer_num):
        dict_a = 'W' + str(i)
        layer = layers[i]
        if layer.type == 'fc':
            mat_shape = layer.W.get_value().shape
            f = path + "/" + os.path.split(nnet_dict[dict_a])[1]
            layer.W.set_value(factor * np.asarray(bp.unpack_ndarray_file(f), dtype=theano.config.floatX).reshape(mat_shape))
        elif layer.type == 'conv':
            filter_shape = layer.filter_shape
            W_array = layer.W.get_value()
            for next_X in range(filter_shape[0]):
                for this_X in range(filter_shape[1]):
                    new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                    mat_shape = W_array[next_X, this_X, :, :].shape
                    f = path + "/" + os.path.split(nnet_dict[new_dict_a])[1]
                    W_array[next_X, this_X, :, :] = factor * np.asarray(bp.unpack_ndarray_file(f), dtype=theano.config.floatX).reshape(mat_shape)
            layer.W.set_value(W_array)
        dict_a = 'b' + str(i)
        f = path + "/" + os.path.split(nnet_dict[dict_a])[1]
        layer.b.set_value(np.asarray(bp.unpack_ndarray_file(f), dtype=theano.config.floatX))
	
def _cnn2file(conv_layers, filename='nnet.out', input_factor = 1.0, factor=[]):
    n_layers = len(conv_layers)
    nnet_dict = {}
    for i in range(n_layers):
       conv_layer = conv_layers[i]
       filter_shape = conv_layer.filter_shape

       dropout_factor = 0.0
       if i == 0:
           dropout_factor = input_factor
       if i > 0 and len(factor) > 0:
           dropout_factor = factor[i-1]

       for next_X in range(filter_shape[0]):
           for this_X in range(filter_shape[1]):
               dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
               nnet_dict[dict_a] = array_2_string(dropout_factor * (conv_layer.W.get_value())[next_X, this_X])

       dict_a = 'b' + str(i)
       nnet_dict[dict_a] = array_2_string(conv_layer.b.get_value())

    with smart_open(filename, 'w') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _file2cnn(conv_layers, filename='nnet.in', factor=1.0):
    n_layers = len(conv_layers)
    nnet_dict = {}

    with smart_open(filename, 'r') as fp:
        nnet_dict = json.load(fp)
    for i in range(n_layers):
        conv_layer = conv_layers[i]
        filter_shape = conv_layer.filter_shape
        W_array = conv_layer.W.get_value()

        for next_X in range(filter_shape[0]):
            for this_X in range(filter_shape[1]):
                dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
                W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[dict_a]))

        conv_layer.W.set_value(W_array)

        dict_a = 'b' + str(i)
        conv_layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
