import theano
import numpy as np

## Softmax function with temperature parameter
def softmax_with_temp(x):
    Temp = 2.0
    e_x = np.exp((x - x.max(axis=1, keepdims=True))/Temp)
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out
