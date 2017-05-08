import theano
import theano.tensor as T
from softmax_with_temp import softmax_with_temp

def custom_crossentropy(combined_targets, nn_output):
    Lambda = 0.8
    _EPS = 10e-8
    hard_targets = T.floor(combined_targets)
    soft_targets = combined_targets - hard_targets
    nn_output_1 = T.nnet.softmax(nn_output)
    nn_output_2 = softmax_with_temp(nn_output)
    nn_output_1 = T.clip(nn_output_1, _EPS, 1.0 - _EPS)
    nn_output_2 = T.clip(nn_output_2, _EPS, 1.0 - _EPS)
    return -((1.0-Lambda) * T.sum(hard_targets * T.log(nn_output_1), axis=1)
           + Lambda * T.sum(soft_targets * T.log(nn_output_2), axis=1))

