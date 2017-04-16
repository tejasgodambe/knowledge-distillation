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
1. run.sh  
This is the master script. 
It takes dnn.nnet.bak file and produces dnn.nnet.svd file. 
Sample dnn.nnet.bak and dnn.nnet.svd files are provided for reference.
Set the values for the identifier num_layers, 1st and 2nd args to block 
function, 1st arg for line function according to your dnn.nnet.bak file. 
The 1st and 2nd args to the block function and 3rd arg to the line
function in the provided run.sh script corresponds to the sample dnn.nnet.bak file.

2. svd.py  
This python script is called by run.sh. It performs the actual SVD operation to
decompose a layer into two sub-layers. 

3. svd_post_formatting.sh  
This BASH script is called by run.sh. It does the post formatting to 
construct the final dnn.nnet.svd file for us. 
