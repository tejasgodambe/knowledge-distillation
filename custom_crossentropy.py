import theano
import theano.tensor as T

# Taken from https://arxiv.org/pdf/1503.02531.pdf
# This crossentropy fn. is a weighted average of two different objective fns. 
# The first objective function is the crossentropy with the soft targets 
# and this cross entropy is computed using the same high temperature in
# the softmax of the distilled model as was used for generating the soft targets# from the cumbersome model. The second objective function is the cross entropy # with the correct labels. This is computed using exactly the same logits in 
# softmax of the distilled model but at a temperature of 1.

def custom_crossentropy(combined_targets, nn_output):
    Lambda = 0.5
    _EPS = 10e-8
    hard_targets = T.floor(combined_targets)
    soft_targets = combined_targets - hard_targets
    nn_output_1 = T.nnet.softmax(nn_output)
    nn_output_2 = softmax_with_temp(nn_output)
    nn_output_1 = T.clip(nn_output_1, _EPS, 1.0 - _EPS)
    nn_output_2 = T.clip(nn_output_2, _EPS, 1.0 - _EPS)
    return -((1.0-Lambda) * T.sum(hard_targets * T.log(nn_output_1), axis=1)
           + Lambda * T.sum(soft_targets * T.log(nn_output_2), axis=1))

