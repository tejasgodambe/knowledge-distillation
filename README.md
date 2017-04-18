# Knowledge distillation
This code is the implementation of the paper "Distilling the knowledge
in a Neural Network" (https://arxiv.org/pdf/1503.02531.pdf).

# Abstract
A very simple way to improve the performance of almost any machine learning
algorithm is to train many different models on the same data and then to average
their predictions [3]. Unfortunately, making predictions using a whole ensemble
of models is cumbersome and may be too computationally expensive to allow deployment
to a large number of users, especially if the individual models are large
neural nets. Caruana and his collaborators [1] have shown that it is possible to
compress the knowledge in an ensemble into a single model which is much easier
to deploy and we develop this approach further using a different compression
technique. We achieve some surprising results on MNIST and we show that we
can significantly improve the acoustic model of a heavily used commercial system
by distilling the knowledge in an ensemble of models into a single model. We also
introduce a new type of ensemble composed of one or more full models and many
specialist models which learn to distinguish fine-grained classes that the full models
confuse. Unlike a mixture of experts, these specialist models can be trained
rapidly and in parallel. 

# Scripts
1. train_student_DNN.py   
This Python script is the main script. In this script, user has to provide the input directories 
such as alignments, HMM, data folder. The output directory is where the weights (after 
each epoch) and the final weights (both in hdf5 and txt format) are saved. 
In this script, we define the DNN architecture and also set the DNN configuration params. 

2. dataGenerator_teacher.py  
This Python script is the generator which provides batches to Keras' fit_generator while training 
teacher DNN. 

3. dataGenerator_student.py  
This Python script is the generator which provides batches to Keras' fit_generator while training
student DNN. 

4. custom_crossentropy.py  
This Python script has the custom crossentropy loss used for training student DNN. 

5. softmax_with_temp.py  
This Python script has the implementation of softmax fn with temperature parameter. 

6. saveModel.py  
This Python script converts DNN weights from hdf5 to txt format.  
