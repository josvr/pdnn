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

import theano
from io_func.model_io import log
import theano.tensor as T
from io_func.data_io import read_data_args, read_dataset
from .learn_rates import LearningRateExpDecay
from .utils import parse_lrate, parse_activation, parse_conv_spec, activation_to_txt, string2bool

class NetworkConfig():
    def totalNumerOfLayers(self): 
        x = 0
        if self.n_outs > 0:
            x = x + 1
        x += len( self.hidden_layers_sizes)
        return x;

    def __init__(self):

        self.model_type = 'DNN'

        self.batch_size = 256
        self.momentum = 0.5
        self.lrate = LearningRateExpDecay(momentum=0.5,start_rate=0.08, scale_by = 0.5,
                                          min_derror_decay_start = 0.05,
                                          min_derror_stop = 0.05,
                                          min_epoch_decay_start=15)

        self.activation = T.nnet.sigmoid
        self.activation_text = 'sigmoid'
        self.do_maxout = False
        self.pool_size = 1

        self.do_dropout = False
        self.dropout_factor = []
        self.input_dropout_factor = 0.0

        self.max_col_norm = None
        self.l1_reg = None
        self.l2_reg = None

        # data reading
        self.train_sets = None
        self.train_xy = None
        self.train_x = None
        self.train_y = None

        self.valid_sets = None
        self.valid_xy = None
        self.valid_x = None
        self.valid_y = None

        self.test_sets = None
        self.test_xy = None
        self.test_x = None
        self.test_y = None

        self.interrupt_epoch = None

        # specifically for DNN
        self.n_ins = 0
        self.hidden_layers_sizes = []
        self.n_outs = 0
        self.non_updated_layers = []

        # specifically for DNN_SAT
        self.ivec_n_ins = 0
        self.ivec_hidden_layers_sizes = []
        self.ivec_n_outs = 0

        # specifically for convolutional networks
        self.conv_layer_configs = []
        self.conv_activation = T.nnet.sigmoid
        self.conv_activation_text = 'sigmoid'
        self.use_fast = False

        # number of epochs between model saving (for later model resuming)
        self.model_save_step = 1

        # the path to save model into Kaldi-compatible format
        self.cfg_output_file = ''
        self.param_output_file = ''
        self.kaldi_output_file = ''

    # initialize pfile reading. TODO: inteference *directly* for Kaldi feature and alignment files
    def init_data_reading(self, train_data_spec, valid_data_spec):
        train_dataset, train_dataset_args = read_data_args(train_data_spec)
        valid_dataset, valid_dataset_args = read_data_args(valid_data_spec)
        self.train_sets, self.train_xy, self.train_x, self.train_y = read_dataset(train_dataset, train_dataset_args)
        self.valid_sets, self.valid_xy, self.valid_x, self.valid_y = read_dataset(valid_dataset, valid_dataset_args)

    def init_data_reading_test(self, data_spec):
        dataset, dataset_args = read_data_args(data_spec)
        self.test_sets, self.test_xy, self.test_x, self.test_y = read_dataset(dataset, dataset_args)

    # initialize the activation function
    def init_activation(self):
        self.activation = parse_activation(self.activation_text)

    def parse_config_common(self, arguments):
        # parse batch_size, momentum, learning rate and regularization
        if 'batch_size' in arguments:
            self.batch_size = int(arguments['batch_size'])
        if 'momentum' in arguments:
            self.momentum = float(arguments['momentum'])
        if 'lrate' in arguments:
            self.lrate = parse_lrate(arguments['lrate'])
        if 'l1_reg' in arguments:
            self.l1_reg = float(arguments['l1_reg'])
        if 'l2_reg' in arguments and str(arguments['l2_reg']).strip().lower() != 'default':
            self.l2_reg = float(arguments['l2_reg'])
        if 'max_col_norm' in arguments:
            self.max_col_norm = float(arguments['max_col_norm'])

        # parse activation function, including maxout
        if 'activation' in arguments:
            self.activation_text = arguments['activation']
            self.activation = parse_activation(arguments['activation'])
            if arguments['activation'].startswith('maxout'):
                self.do_maxout = True
                self.pool_size = int(arguments['activation'].replace('maxout:',''))
                self.activation_text = 'maxout'

        # parse dropout. note that dropout can be applied to the input features only when dropout is also
        # applied to hidden-layer outputs at the same time. that is, you cannot apply dropout only to the
        # input features
        if 'dropout_factor' in arguments and str(arguments['dropout_factor']).strip().lower() != 'default':
            self.do_dropout = True
            factors = arguments['dropout_factor'].split(',')
            self.dropout_factor = [float(factor) for factor in factors]
            if 'input_dropout_factor' in arguments  and str(arguments['input_dropout_factor']).strip().lower() != 'default':
                self.input_dropout_factor = float(arguments['input_dropout_factor'])

        if 'cfg_output_file' in arguments:
            self.cfg_output_file = arguments['cfg_output_file']
        if 'param_output_file' in arguments:
            self.param_output_file = arguments['param_output_file']
        if 'kaldi_output_file' in arguments:
            self.kaldi_output_file = arguments['kaldi_output_file']

        if 'model_save_step' in arguments:
            self.model_save_step = int(arguments['model_save_step'])

        if 'non_updated_layers' in arguments:
            layers = arguments['non_updated_layers'].split(",")
            self.non_updated_layers = [int(layer) for layer in layers]

    def parse_config_dnn(self, arguments, nnet_spec):
        self.parse_config_common(arguments)
        # parse DNN network structure
        if 'interrupt_after_epoch' in arguments:
           self.interrupt_epoch = int(arguments['interrupt_after_epoch'])
        nnet_layers = nnet_spec.split(':')
        self.n_ins = int(nnet_layers[0])
        self.hidden_layers_sizes = [int(nnet_layers[i]) for i in range(1, len(nnet_layers)-1)]
        self.n_outs = int(nnet_layers[-1])
        log(">> DNN Config parsed. Start Dump")
        log("interrupt_epoch="+str(self.interrupt_epoch))
        log("batch_size="+str(self.batch_size))
        log("momentum="+str(self.momentum ))
        log("lrate="+str(self.lrate ))
        log("activation="+str(self.activation))
        log("activation_text="+str(self.activation_text))
        log("do_maxout="+str(self.do_maxout))
        log("pool_size="+str(self.pool_size))
        log("do_dropout="+str(self.do_dropout))
        log("dropout_factor="+str(self.dropout_factor))
        log("input_dropout_factor="+str(self.input_dropout_factor))
        log("max_col_norm="+str(self.max_col_norm))
        log("l1_reg="+str(self.l1_reg))
        log("l2_reg="+str(self.l2_reg))
        log("n_ins="+str(self.n_ins))
        log("hidden_layers_sizes="+str(self.hidden_layers_sizes))
        log("n_outs="+str(self.n_outs))
        log("non_updated_layers="+str(self.non_updated_layers))
        log("use_fast="+str(self.use_fast))
        log("model_save_step="+str(self.model_save_step))
        log("cfg_output_file="+str(self.cfg_output_file)) 
        log("param_output_file="+str(self.param_output_file)) 
        log("kaldi_output_file="+str(self.kaldi_output_file))
        log("<< DNN Config parsed. End Dump")

    def parse_config_cnn(self, arguments, nnet_spec, conv_nnet_spec):
        self.parse_config_dnn(arguments, nnet_spec)
        # parse convolutional layer structure
        self.conv_layer_configs = parse_conv_spec(conv_nnet_spec, self.batch_size)
        # parse convolutional layer activation
        # parse activation function, including maxout
        if 'conv_activation' in arguments:
            self.conv_activation_text = arguments['conv_activation']
            self.conv_activation = parse_activation(arguments['conv_activation'])
            # maxout not supported yet
        # whether we use the fast version of convolution
        if 'use_fast' in arguments:
            self.use_fast = string2bool(arguments['use_fast'])
        log(">> CNN Config parsed. Start Dump")
        log("batch_size="+str(self.batch_size))
        log("momentum="+str(self.momentum ))
        log("lrate="+str(self.lrate ))
        log("activation="+str(self.activation))
        log("activation_text="+str(self.activation_text))
        log("do_maxout="+str(self.do_maxout))
        log("pool_size="+str(self.pool_size))
        log("do_dropout="+str(self.do_dropout))
        log("dropout_factor="+str(self.dropout_factor))
        log("input_dropout_factor="+str(self.input_dropout_factor))
        log("max_col_norm"+str(self.max_col_norm))
        log("l1_reg="+str(self.l1_reg))
        log("l2_reg="+str(self.l2_reg))
        log("n_ins="+str(self.n_ins))
        log("hidden_layers_sizes="+str(self.hidden_layers_sizes))
        log("n_outs="+str(self.n_outs))
        log("non_updated_layers="+str(self.non_updated_layers))
        log("ivec_n_ins="+str(self.ivec_n_ins))
        log("ivec_hidden_layers_sizes="+str(self.ivec_hidden_layers_sizes))
        log("ivec_n_outt="+str(self.ivec_n_outs))
        log("conv_layer_configs="+str(self.conv_layer_configs))
        log("conv_activation="+str(self.conv_activation))
        log("conv_activation_tex="+str(self.conv_activation_text))
        log("use_fast="+str(self.use_fast))
        log("model_save_step="+str(self.model_save_step))
        log("cfg_output_file="+str(self.cfg_output_file)) 
        log("param_output_file="+str(self.param_output_file)) 
        log("kaldi_output_file="+str(self.kaldi_output_file))
        log("<< CNN Config parsed. End Dump")
