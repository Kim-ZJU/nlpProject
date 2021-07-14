from collections import Counter
import numpy as np
import random
from collections import OrderedDict
import math

import mindspore
import mindspore.nn as nn
from mindspore import Tensor

#from mindspore.ops import operations as ops
from mindspore import ops
from mindspore import dtype as mstype

floatX = np.float32

def init_weight(n, d, options):
    ''' initialize weight matrix
    options['init_type'] determines
    gaussian or uniform initlizaiton
    '''
    if options['init_type'] == 'gaussian':
        return (np.random.randn(n, d).astype(floatX)) * options['std']
    elif options['init_type'] == 'uniform':
        # [-range, range]
        return ((np.random.rand(n, d) * 2 - 1) * \
                options['range']).astype(floatX)
def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are
    orthogonal.
    """
    W = np.random.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')

def init_fflayer(params, nin, nout, options, prefix='ff'):
    ''' initialize ff layer
    '''
    params[prefix + '_w'] = init_weight(nin, nout, options)
    params[prefix + '_b'] = np.zeros(nout, dtype='float32')
    return params

def init_lstm_layer(params, nin, ndim, options, prefix='lstm'):
    ''' initializt lstm layer
    '''
    params[prefix + '_w_x'] = init_weight(nin, 4 * ndim, options)
    # use svd trick to initializ
    if options['init_lstm_svd']:
        params[prefix + '_w_h'] = np.concatenate([ortho_weight(ndim),
                                                  ortho_weight(ndim),
                                                  ortho_weight(ndim),
                                                  ortho_weight(ndim)],
                                                 axis=1)
    else:
        params[prefix + '_w_h'] = init_weight(ndim, 4 * ndim, options)
    params[prefix + '_b_h'] = np.zeros(4 * ndim, dtype='float32')
    # set forget bias to be positive
    params[prefix + '_b_h'][ndim : 2*ndim] = np.float32(options.get('forget_bias', 0))
    return params

# initialize the parmaters
def init_params(options):
    ''' Initialize all the parameters
    '''
    params = OrderedDict()
    n_words = options['n_words']
    n_emb = options['n_emb']
    n_dim = options['n_dim']
    n_image_feat = options['n_image_feat']
    n_common_feat = options['n_common_feat']
    n_output = options['n_output']
    n_attention = options['n_attention']

    params['w_emb'] = Tensor((np.random.rand(n_words, n_emb) * 2 - 1) * 0.5).astype(floatX)

    params = init_fflayer(params, n_image_feat, n_dim, options,
                          prefix='image_mlp')

    # attention model based parameters
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='image_att_mlp_1')
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='sent_att_mlp_1')
    params = init_fflayer(params, n_attention, 1, options,
                          prefix='combined_att_mlp_1')
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='image_att_mlp_2')
    params = init_fflayer(params, n_dim, n_attention, options,
                          prefix='sent_att_mlp_2')
    params = init_fflayer(params, n_attention, 1, options,
                          prefix='combined_att_mlp_2')


    # params for sentence image mlp
    for i in range(options['combined_num_mlp']):
        if i == 0 and options['combined_num_mlp'] == 1:
            params = init_fflayer(params, n_dim, n_output,
                                  options, prefix='combined_mlp_%d'%(i))
        elif i == 0 and options['combined_num_mlp'] != 1:
            params = init_fflayer(params, n_dim, n_common_feat,
                                  options, prefix='combined_mlp_%d'%(i))
        elif i == options['combined_num_mlp'] - 1 :
            params = init_fflayer(params, n_common_feat, n_output,
                                  options, prefix='combined_mlp_%d'%(i))
        else:
            params = init_fflayer(params, n_common_feat, n_common_feat,
                                  options, prefix='combined_mlp_%d'%(i))

    # lstm layer
    params = init_lstm_layer(params, n_emb, n_dim, options, prefix='sent_lstm')

    return params

def init_shared_params(params):
    ''' return a shared version of all parameters
    '''
    global shared_params
    shared_params = OrderedDict()
    for k, p in params.items():
        shared_params[k] = params[k]

    return shared_params

def fflayer(shared_params, x, options, prefix='ff', act_func='tanh'):
    ''' fflayer: multiply weight then add bias
    '''
    tanh = nn.Tanh()
    input = ops.dot(x, Tensor(shared_params[prefix + '_w'])) + \
                          Tensor(shared_params[prefix + '_b'])
    return tanh(input)

def fflayer1(shared_params, x, options, prefix='ff'):
    ''' fflayer: multiply weight then add bias
    '''
    input = ops.dot(x, Tensor(shared_params[prefix + '_w'])) + \
                          Tensor(shared_params[prefix + '_b'])
    return input

def get_lr(options, curr_epoch):
    if options['optimization'] == 'sgd':
        power = max((curr_epoch - options['step_start']) / options['step'], 0)
        power = math.ceil(power)
        return options['lr'] * (options['gamma'] ** power)  #
    else:
        return options['lr']