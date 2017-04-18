#!/usr/bin/python3

import keras
import numpy
import sys

def saveModel (model, fileName):
    with open (fileName, 'w') as f:
        for layer in model.layers:
            if layer.name.startswith ('dense') or layer.name.startswith ('nondense'):
                f.write ('<affinetransform> %d %d\n[\n' % (layer.output_shape[1], layer.input_shape[1]))
                for row in layer.get_weights()[0].T:
                    row.tofile (f, format="%e", sep=' ')
                    f.write ('\n')
                f.write (']\n[ ')
                layer.get_weights()[1].tofile (f, format="%e", sep=' ')
                f.write (' ]\n')
                activation_text = layer.get_config()['activation']
                #if activation_text != 'linear':
                f.write ('<%s> %d %d\n' % (activation_text, layer.output_shape[1], layer.output_shape[1]))
            elif layer.name.startswith ('maxoutdense'):
                f.write ('<affinetransform> %d %d\n[\n' % (layer.nb_feature * layer.output_shape[1], layer.input_shape[1]))
                for row in numpy.hstack(layer.get_weights()[0].T).T:
                    row.tofile (f, format="%e", sep=' ')
                    f.write ('\n')
                f.write (']\n[ ')
                numpy.hstack(layer.get_weights()[1].T).T.tofile (f, format="%e", sep=' ')
                f.write (' ]\n')
                f.write ('<maxout> %d %d\n' % (layer.output_shape[1], layer.nb_feature * layer.output_shape[1]))
            elif layer.name.startswith ('activation'):
                activation_text = layer.get_config()['activation']
                f.write ('<%s> %d %d\n' % (activation_text, layer.output_shape[1], layer.input_shape[1]))
            elif layer.name.startswith ('dropout'):
                pass
            else:
                raise TypeError ('Unknown layer type: ' + layer.name)

## Save h5 model in nnet format
if __name__ == '__main__':
    h5model = sys.argv[1]
    nnet = sys.argv[2]
    m = keras.models.load_model (h5model)
    saveModel(m, nnet)
