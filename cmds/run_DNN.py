# Copyright 2014    Yajie Miao         Carnegie Mellon University
# Copyright 2016    Jos van Roosmalen  Open University The Netherlands

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

import pickle
import gzip
import os
import os.path
import sys
import time
import shutil

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from models.dnn import DNN
from models.dropout_nnet import DNN_Dropout

from io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from utils.utils import parse_arguments
from utils.learn_rates import _lrate2file, _file2lrate

from utils.network_config import NetworkConfig
from learning.sgd import train_sgd, validate_by_minibatch
from utils.stop_handler import stop_if_stop_is_requested;

def saveModel(dnn,cfg):
    log("> Start saveModel")
    # save the model and network configuration
    if cfg.param_output_file != '':
        _nnet2file(dnn.layers, path=cfg.param_output_file, input_factor = cfg.input_dropout_factor, factor = cfg.dropout_factor)
        log('> ... the best PDNN model param so far is ' + cfg.param_output_file)
    if cfg.cfg_output_file != '':
        _cfg2file(dnn.cfg, filename=cfg.cfg_output_file)
        log('> ... the best PDNN model config so far is ' + cfg.cfg_output_file)

    # output the model into Kaldi-compatible format
    if cfg.kaldi_output_file != '':
        dnn.write_model_to_kaldi(cfg.kaldi_output_file)
        log('> ... the best Kaldi model so far is ' + cfg.kaldi_output_file)
    log("< End SaveModel")

if __name__ == '__main__':
    stop_if_stop_is_requested()
    log('Run DNN')
    log('Theano Config:')
    log(theano.config)
    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'valid_data', 'nnet_spec', 'wdir']
    for arg in required_arguments:
        if (arg in arguments) == False:
            print("Error: the argument %s has to be specified" % (arg)); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig()
    cfg.parse_config_dnn(arguments, nnet_spec)
    cfg.init_data_reading(train_data_spec, valid_data_spec)

    # parse pre-training options
    # pre-training files and layer number (how many layers are set to the pre-training parameters)
    ptr_layer_number = 0; ptr_file = ''
    if 'ptr_file' in arguments and 'ptr_layer_number' in arguments:
        ptr_file = arguments['ptr_file']
        if ptr_file.lower() == 'default': 
            ptr_file = ''
        ptr_layer_number = int(arguments['ptr_layer_number'])
 
    log('Ptr-file "'+ptr_file+'" ptr layer='+str(ptr_layer_number));

    # check working dir to see whether it's resuming training
    resume_training = False
    if os.path.exists(wdir + '/dnn.tmp') and os.path.exists(wdir + '/dnn_training_state.tmp'):
        resume_training = True
        cfg.lrate = _file2lrate(wdir + '/dnn_training_state.tmp')
        log('> ... found nnet.tmp and training_state.tmp, now resume training from epoch ' + str(cfg.lrate.epoch))

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... building the model')
    # setup model
    if cfg.do_dropout:
        dnn = DNN_Dropout(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
    else:
        dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)

    # initialize model parameters
    # if not resuming training, initialized from the specified pre-training file
    # if resuming training, initialized from the tmp model file
    if (ptr_file != '' and ptr_layer_number != 0) and (resume_training is False):
        log('>... Loading pretrained data from '+str(ptr_file))
        _file2nnet(dnn.layers, set_layer_num = ptr_layer_number, path = ptr_file)
    if resume_training:
        _file2nnet(dnn.layers, path = wdir + '/dnn.tmp')

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y),
                batch_size=cfg.batch_size)
    lowest_validation_error = None
    fail_count = 0
    log('> ... finetuning the model')
    while (cfg.lrate.get_rate() != 0):
        stop_if_stop_is_requested()
        # one epoch of sgd training 
        train_error = train_sgd(train_fn, cfg)
        log('> epoch %d, training error %f ' % (cfg.lrate.epoch, 100*numpy.mean(train_error)) + '(%)')
        # validation 
        valid_error = validate_by_minibatch(valid_fn, cfg)
        valid_percent = 100*numpy.mean(valid_error)
        msg = ""
        if lowest_validation_error is None:
            lowest_validation_error = valid_percent
            msg += "(new low)"
            fail_count = 0
            saveModel(dnn,cfg)
        else:
            if valid_percent < lowest_validation_error:
                msg += "(new low)"
                lowest_validation_error = valid_percent
                fail_count = 0
                saveModel(dnn,cfg)
            else:
                fail_count += 1
                msg += "(failed count for "+str(fail_count)+")"

        log('> epoch %d, lrate %f, validation error %f ' % (cfg.lrate.epoch, cfg.lrate.get_rate(), valid_percent) + '(%) '+msg)
        if cfg.interrupt_epoch != None and cfg.lrate.epoch == cfg.interrupt_epoch: 
            log("** GOING TO INTERRUPT as requested")
            sys.exit(0)
        cfg.lrate.get_next_rate(current_error = valid_percent )
 
        # output nnet parameters and lrate, for training resume
        if cfg.lrate.epoch % cfg.model_save_step == 0:
            _nnet2file(dnn.layers, path=wdir + '/dnn.tmp')
            _lrate2file(cfg.lrate, wdir + '/dnn_training_state.tmp') 

    
    # remove the tmp files (which have been generated from resuming training) 
    if os.path.exists(wdir + '/dnn.tmp'):
        shutil.rmtree(wdir + '/dnn.tmp')
    if os.path.exists(wdir + '/dnn_training_state.tmp'):
        os.remove(wdir + '/dnn_training_state.tmp') 


