import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

def adam(loss, all_params, dparams,t,m,v,learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = OrderedDict()
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g,dparam,m_previous,v_previous in zip(all_params, all_grads,dparams,m,v):
        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        step = - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates[m_previous] = m
        updates[v_previous] = v
        updates[dparam] = step
    updates[t]= t + 1.
    return updates
