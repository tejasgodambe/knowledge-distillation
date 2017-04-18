#!/usr/bin/python3

# Author: Tejas Godambe 
# Date: April 2017

import sys, os, time
import numpy as np
import theano
import theano.tensor as T 
import keras
from subprocess import Popen, PIPE, DEVNULL
from dataGenerator_student import dataGenerator_student_tr
from dataGenerator_student import dataGenerator_student_cv
import custom_crossentropy, softmax_with_temp, kaldiIO, saveModel
from average_DNN_predictions import geometric_mean, arithmetic_mean

############# USER CONFIGURABLE PARAMS STARTS #############
## Learning parameters
learning = {'rate' : 0.08,
            'batchSize' : 256}

## Input directories
data_tr = 'train_tr95_20hrs_first35kUtts'
data_cv = 'train_cv05'
sgmm='/home/pavan/2016h2/en-US-wc-sou-dd2/exp/sgmm2_4a_1500'
ali_tr='/home/pavan/2016h2/en-US-wc-sou-dd2/exp/sgmm2_4a_1500_ali_tr95'
ali_cv='/home/pavan/2016h2/en-US-wc-sou-dd2/exp/sgmm2_4a_1500_ali_cv05'

## Output directories
outdir = 'outdir_en-US_273_256_256_1375_20hrs_first35kUtts_student_Lambda_0.5_0.5'
model_out = outdir + '/' + 'dnn.nnet'
model_out_h5 = model_out + '.h5'
############# USER CONFIGURABLE PARAMS ENDS #############

## Main code
if '__name__' == '__main__': 

    os.makedirs (exp, exist_ok=True)
    
    logfile = sys.argv[0] + '.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    log_obj = open (logfile, 'w')
    sys.stdout = log_obj
    
    model_list = ['outdir_en-US_273_512_512_1375_20hrs_first35kUtts_teacher/dnn.nnet.h15']
    model_list = [keras.models.load_model(m) for m in model_list]
    weights = list(np.ones(len(model_list)))

    ## Save teacher predictions on disk    
    st = time.time()
    outfile = data_tr + '/teacher_predictions.ark'
    print ('Writing teacher predictions...', outfile)
    p1 = Popen (['splice-feats','--print-args=false','--left-context=3','--right-context=3',
             'scp:' + data_tr + '/feats.scp','ark:-'], stdout=PIPE)
    with open ('temp_teacher_dummyExpt.ark', 'wb') as f:
        while True:
            uid, featMat = kaldiIO.readUtterance (p1.stdout)
            if uid == None:
                break
            avg_prediction = geometric_mean(featMat, model_list, weights)
            kaldiIO.writeUtterance(uid, avg_prediction, f, sys.stdout.encoding)
    p1.stdout.close()
    et = time.time()
    os.rename('temp_teacher_dummyExpt.ark', outfile)
    print ('Done writing teacher predictions for ', data_tr, '. Time taken: ', et-st)
    
    ## Initialize data generator
    trGen = dataGenerator_7window_student_tr (data_tr, ali_tr, sgmm, learning['batchSize'])
    cvGen = dataGenerator_7window_student_cv (data_cv, ali_cv, sgmm, learning['batchSize'])
    
    ## Define DNN architecture and initialize weights
    m = keras.models.Sequential([
                    keras.layers.Dense(256, activation='relu', input_dim=273),
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(1375, activation='linear')])
    
    print (m.summary())
    
    ## Configure DNN training
    s = keras.optimizers.SGD(lr=learning['rate'], decay=0, momentum=0.5, nesterov=True)
    m.compile(loss=custom_crossentropy, optimizer=s, metrics=['accuracy'])
    
    print ('Learning rate: %f' % learning['rate'])
    
    ## Keras callbacks
    filepath = exp + '/weights.{epoch:02d}-{val_acc:.2f}.h5'
    # to save weights after each epoch
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', 
                   verbose=1, save_best_only=False, mode='auto')
    # to reduce learning rate if loss marginally decreases for consecutive epochs 
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, 
                patience=1, epsilon=0.25, mode='auto', verbose=1, min_lr=0)
    # to stop training before overfitting happens
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
                     min_delta=0.005, patience=3, verbose=1, mode='auto')
    
    ## DNN training 
    m.fit_generator (trGen, samples_per_epoch=trGen.numFeats,
          validation_data=cvGen, nb_val_samples=cvGen.numFeats,
          nb_epoch=30, verbose=1, callbacks=[checkpointer, reduce_lr, early_stopping]) 
    
    ## Save final weights
    m.save_weights (retrained_model_h5, overwrite=True)
    saveModel (m, retrained_model) # convert weights in h5 to txt format
    
    print ('Learning finished ...')
