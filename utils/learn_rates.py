# Copyright 2014    Yajie Miao    Carnegie Mellon University

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

import pickle
import numpy as np
import theano
import theano.tensor as T
import collections
from utils.adam import adam

from io_func.model_io import log

from io_func import smart_open

class LearningRate(object):

    def __init__(self):
        '''constructor'''

    def save(self):
        return None

    def resume(self,input_value):
        pass

    def get_rate(self):
        pass

    def get_next_rate(self, current_error):
        pass

class LearningRateAdam(LearningRate):
    def __init__(self,thres_fail = 1.00,max_fail=6,max_epoch=100,learning_rate=0.001, beta1=0.9,beta2=0.999, epsilon=1e-8,gamma=1-1e-8):
        log("Init Adam with thres_fail="+str(thres_fail)+" max_fail="+str(max_fail)+" max_epoch="+str(max_epoch)+" learning_rate="+str(learning_rate)+" beta1="+str(beta1)+" beta2="+str(beta2)+" epsilon="+str(epsilon)+" gamma="+str(gamma))
        self.learning_rate =  theano.shared(np.asarray(learning_rate, dtype=theano.config.floatX))
        self.beta1 =  theano.shared(np.asarray(beta1, dtype=theano.config.floatX))
        self.beta2 =  theano.shared(np.asarray(beta2, dtype=theano.config.floatX))
        self.epsilon =  theano.shared(np.asarray(epsilon, dtype=theano.config.floatX))
        self.gamma = theano.shared(np.asarray(gamma,dtype=theano.config.floatX))
        self.max_fail = max_fail
        self.max_epoch = max_epoch
        self.thres_fail = thres_fail
        self.lowest_error = None
        self.epoch = 1
        self.rate = 1
        self.prev_error = None
        self.fails = 0
        self.do_resume = False

    def get_rate(self):
        return self.rate

    def save(self): 
        log("Save state adam" )
        m = [x.get_value() for x in self.m_previous]
        v = [x.get_value() for x in self.v_previous]
        t = self.t.get_value()
        return (t,m,v)

    def resume(self,obj): 
        log("Resume state adam") 
        self.resume_t = obj[0]
        self.resume_m = obj[1]
        self.resume_v = obj[2]
        self.do_resume = True
   
    def getOptimizerUpdates(self,cost,delta_params,params):
        self.t = theano.shared(np.asarray(1., dtype=theano.config.floatX))
        self.m_previous = [theano.shared(x.get_value() * 0.) for x in params] 
        self.v_previous = [theano.shared(x.get_value() * 0.) for x in params]
        if self.do_resume: 
            log("Resume settings adam") 
            self.t.set_value(self.resume_t)
            for x,x1 in zip(self.m_previous,self.resume_m):
                x.set_value(x1)
            for x,x1 in zip(self.v_previous,self.resume_v):
                x.set_value(x1)
  
        return adam(cost,params,delta_params,self.t,self.m_previous,self.v_previous,self.learning_rate,self.beta1,self.beta2,self.epsilon,self.gamma)

    def get_next_rate(self, current_error):
        if self.epoch >= self.max_epoch:
            self.rate = 0.0

        if self.lowest_error is None:
           self.lowest_error = current_error;
           self.fails = 0
        else:
          if current_error >= self.lowest_error * self.thres_fail:
              self.fails += 1
              if self.fails >= self.max_fail:
                 self.rate = 0.0
          else:
              if current_error < self.lowest_error * 0.995:
                 self.lowest_error = current_error
                 self.fails = 0
              else:
                 self.fails += 1
                 if self.fails >= self.max_fail:
                      self.rate = 0.0

        self.epoch += 1
        self.prev_error = current_error
        return self.rate

class LearningRateConstant(LearningRate):

    def __init__(self, momentum,learning_rate = 0.08, epoch_num = 20):

        self.learning_rate = learning_rate
        self.epoch = 1
        self.epoch_num = epoch_num
        self.rate = learning_rate
        self.momentum=theano.shared(np.asarray(momentum, dtype=theano.config.floatX))
        self.tRate = theano.shared(np.asarray(self.rate, dtype=theano.config.floatX))

    def getOptimizerUpdates(self,cost,delta_params,params):
        updates = collections.OrderedDict()
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, params)
        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(delta_params, gparams):
            updates[dparam] = self.momentum * dparam - gparam*self.tRate

        return updates;

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):

        if ( self.epoch >=  self.epoch_num):
            self.rate = 0.0
        else:
            self.rate = self.learning_rate
        self.epoch += 1
        self.tRate.set_value(self.rate)
        return self.rate

class LearningRateExpDecay(LearningRate):

    def __init__(self, momentum=0.5,start_rate = 0.08, scale_by = 0.5,
                 min_derror_decay_start = 0.05, min_derror_stop = 0.05, init_error = 100,
                 decay=False, min_epoch_decay_start=15, zero_rate = 0.0):

        self.momentum=theano.shared(np.asarray(momentum, dtype=theano.config.floatX))
        self.tRate = theano.shared(np.asarray(start_rate, dtype=theano.config.floatX))

        self.start_rate = start_rate
        self.init_error = init_error

        self.rate = start_rate
        self.scale_by = scale_by
        self.min_derror_decay_start = min_derror_decay_start
        self.min_derror_stop = min_derror_stop
        self.lowest_error = init_error

        self.epoch = 1
        self.decay = decay
        self.zero_rate = zero_rate

        self.min_epoch_decay_start = min_epoch_decay_start

    def getOptimizerUpdates(self,cost,delta_params,params):
        updates = collections.OrderedDict()
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, params)
        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(delta_params, gparams):
            updates[dparam] = self.momentum * dparam - gparam*self.tRate
      
        return updates

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):
        diff_error = 0.0
        diff_error = self.lowest_error - current_error

        if (current_error < self.lowest_error):
            self.lowest_error = current_error

        if (self.decay):
            if (diff_error < self.min_derror_stop):
                self.rate = 0.0
            else:
                self.rate *= self.scale_by
        else:
            if ((diff_error < self.min_derror_decay_start) and (self.epoch > self.min_epoch_decay_start)):
                self.decay = True
                self.rate *= self.scale_by

        self.epoch += 1
        self.tRate.set_value(self.rate)
        return self.rate


class LearningMinLrate(LearningRate):

    def __init__(self, start_rate = 0.08, scale_by = 0.5,
                 min_derror_decay_start = 0.05,
                 min_lrate_stop = 0.0002, init_error = 100,
                 decay=False, min_epoch_decay_start=15):

        self.start_rate = start_rate
        self.init_error = init_error

        self.rate = start_rate
        self.scale_by = scale_by
        self.min_lrate_stop = min_lrate_stop
        self.lowest_error = init_error

        self.min_derror_decay_start = min_derror_decay_start
        self.epoch = 1
        self.decay = decay
        self.min_epoch_decay_start = min_epoch_decay_start

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):
        diff_error = 0.0

        diff_error = self.lowest_error - current_error

        if (current_error < self.lowest_error):
            self.lowest_error = current_error

        if (self.decay):
            if (self.rate < self.min_lrate_stop):
                self.rate = 0.0
            else:
                self.rate *= self.scale_by
        else:
            if (diff_error < self.min_derror_decay_start) and (self.epoch >= self.min_epoch_decay_start):
                self.decay = True
                self.rate *= self.scale_by

        self.epoch += 1
        return self.rate

class LearningFixedLrate(LearningRate):

    def __init__(self, start_rate = 0.08, scale_by = 0.5,
                 decay_start_epoch = 10, init_error = 100,
                 decay=False, stop_after_deday_epoch=6):

        self.start_rate = start_rate
        self.init_error = init_error

        self.rate = start_rate
        self.scale_by = scale_by
        self.decay_start_epoch = decay_start_epoch
        self.stop_after_deday_epoch = stop_after_deday_epoch
        self.lowest_error = init_error

        self.epoch = 1
        self.decay = decay

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):
        diff_error = 0.0

        diff_error = self.lowest_error - current_error

        if (current_error < self.lowest_error):
            self.lowest_error = current_error

        if (self.decay):
            if (self.epoch >= self.decay_start_epoch + self.stop_after_deday_epoch):
                self.rate = 0.0
            else:
                self.rate *= self.scale_by
        else:
            if (self.epoch >= self.decay_start_epoch):
                self.decay = True
                self.rate *= self.scale_by

        self.epoch += 1
        return self.rate

class LearningRateAdaptive(LearningRate):

    def __init__(self, momentum = 0.5,lr_init = 0.08,
                 thres_inc = 1.00, factor_inc = 1.05,
                 thres_dec = 1.04, factor_dec = 0.7,
                 thres_fail = 1.00, max_fail = 6,
                 max_epoch = 100):
        log("Init LearningRate Adaptive with momentum="+str(momentum)+" lr_init="+str(lr_init)+"  thres_inc="+str( thres_inc)+" factor_inc="+str(factor_inc)+" thres_dec="+str(thres_dec)+" factor_dec="+str(factor_dec)+" thres_fail="+str(thres_fail)+" max_fail="+str(max_fail)+" max_epoch="+str(max_epoch));
        self.rate = lr_init
        self.thres_inc = thres_inc
        self.factor_inc = factor_inc
        self.thres_dec = thres_dec
        self.factor_dec = factor_dec
        self.thres_fail = thres_fail
        self.max_fail = max_fail
        self.max_epoch = max_epoch

        self.lowest_error = None

        self.epoch = 1
        self.prev_error = None
        self.fails = 0
        self.momentum=theano.shared(np.asarray(momentum, dtype=theano.config.floatX))
        self.tRate = theano.shared(np.asarray(self.rate, dtype=theano.config.floatX))

    def getOptimizerUpdates(self,cost,delta_params,params):
        updates = collections.OrderedDict()
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, params)
        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(delta_params, gparams):
            updates[dparam] = self.momentum * dparam - gparam*self.tRate

        return updates;

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):
        log("> get_next_rate:");
        log("PRE: current_error="+str(current_error)+" prev="+str( self.prev_error)+" lowest_error="+str( self.lowest_error)+ " epoch="+str(self.epoch)+" max epoch="+str(self.max_epoch)+" fails="+str(self.fails)+" max fail="+str(self.max_fail))
        if self.epoch >= self.max_epoch:
            self.rate = 0.0
        elif self.prev_error is not None:
           if current_error < self.prev_error * self.thres_inc:
              self.rate *= self.factor_inc
           elif current_error >= self.prev_error * self.thres_dec:
              self.rate *= self.factor_dec

        if self.lowest_error is None:
           self.lowest_error = current_error;
           self.fails = 0
        else:
           if current_error >= self.lowest_error * self.thres_fail:
              self.fails += 1
              if self.fails >= self.max_fail:
                 self.rate = 0.0
           else: 
              if current_error < self.lowest_error * 0.995:
                 self.lowest_error = current_error
                 self.fails = 0
              else:
                 self.fails += 1
                 if self.fails >= self.max_fail:
                      self.rate = 0.0

        self.epoch += 1
        self.prev_error = current_error
        log("POST: rate="+str(self.rate)+" current_error="+str(current_error)+" prev="+str( self.prev_error)+" lowest_error="+str( self.lowest_error)+ " epoch="+str(self.epoch)+" max epoch="+str(self.max_epoch)+" fails="+str(self.fails)+" max fail="+str(self.max_fail))
        log("< get_next_rate")
        self.tRate.set_value(self.rate)
        return self.rate


# save and load the learning rate class
def _lrate2file(lrate, filename='file.out'):
    with smart_open(filename, "wb") as output:
        obj = lrate.save()
        pickle.dump((lrate,obj), output, pickle.HIGHEST_PROTOCOL)

def _file2lrate(filename='file.in'):
    lrate,obj = pickle.load(smart_open(filename,'rb'))
    #log("Lrate ="+str(lrate))
    #log("obj =" +str(obj))
    lrate.resume(obj)
    return lrate


