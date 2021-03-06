#  Copyright    2016    Jos van Roosmalen Open University The Netherlands

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

import os
import sys, re
import glob
import bloscpack as bp
import numpy
import theano
import theano.tensor as T
from utils.utils import string2bool
from .model_io import log
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label

#
# Directly derived (without refactoring ;-)) from the Pickler reader to handle BloscPack files
#

class BloscPackDataRead(object):

    def __init__(self, pfile_path_list, read_opts):

        self.pfile_path_list = pfile_path_list
        self.cur_pfile_index = 0
        self.pfile_path = pfile_path_list[0]
        self.read_opts = read_opts

        self.feat_mat = None
        self.label_vec = None

        # other variables to be consistent with PfileDataReadStream
        self.cur_frame_num = 0
        self.end_reading = False
        self.total_size = None

    def get_total_size(self):
        if self.total_size != None:
            return self.total_size
        self.total_size = 0
        for f in self.pfile_path_list:
            l = bp.unpack_ndarray_file(f+".labels")
            self.total_size += len(l)
        return self.total_size 

    def load_next_partition(self, shared_xy):
        pfile_path = self.pfile_path_list[self.cur_pfile_index]
        if self.feat_mat is None or len(self.pfile_path_list) > 1:
            #log("Start reading partition "+pfile_path) 
            self.feat_mat = bp.unpack_ndarray_file(pfile_path)
            self.label_vec = bp.unpack_ndarray_file(pfile_path+".labels")  
            shared_x, shared_y = shared_xy

            self.feat_mat, self.label_vec = \
                preprocess_feature_and_label(self.feat_mat, self.label_vec, self.read_opts)
            if self.read_opts['random']:
                shuffle_feature_and_label(self.feat_mat, self.label_vec)

            shared_x.set_value(self.feat_mat, borrow=True)
            shared_y.set_value(self.label_vec.astype(theano.config.floatX), borrow=True)
            #log("Finished reading partition "+pfile_path)
        self.cur_frame_num = len(self.feat_mat)
        self.cur_pfile_index += 1

        if self.cur_pfile_index >= len(self.pfile_path_list):   # the end of one epoch
            self.end_reading = True
            self.cur_pfile_index = 0
        return pfile_path

    def is_finish(self):
        return self.end_reading

    def initialize_read(self, first_time_reading = False):
        self.end_reading = False

    def make_shared(self):
        # define shared variables
        feat = numpy.zeros((10,10), dtype=theano.config.floatX)
        label = numpy.zeros((10,), dtype=theano.config.floatX)

        shared_x = theano.shared(feat, name = 'x', borrow = True)
        shared_y = theano.shared(label, name = 'y', borrow = True)
        return shared_x, shared_y

