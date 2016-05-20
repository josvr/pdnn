#!/bin/bash

# two variables you need to set
pdnndir=..  # pointer to PDNN
device=cpu # gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# download mnist dataset
wget -nc http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz	

# split the dataset to training, validation and testing sets
# you will see train.pickle.gz, valid.pickle.gz, test.pickle.gz
echo "Preparing datasets ..." >> nn.log
python3 data_prep.py >> nn.log

rm -rf dnn*

# train DNN model
echo "Training Adam Test ..." >> nn.log
python3 $pdnndir/cmds/run_DNN.py --train-data "train.part*.blp" \
                                --valid-data "valid.part*.blp" \
                                --nnet-spec "784:1024:1024:10" --wdir ./ \
                                --activation tanh \
                                --l2-reg 0.0001 --lrate "ADAM:1.00:6:5:0.0005:0.9:0.999:1e-8:0.99999999" --model-save-step 1 \
                                --param-output-file dnn.param --cfg-output-file dnn.cfg  >> nn.log 2>&1

# classification on the testing data; -1 means the final layer, that is, the classification softmax layer
echo "Classifying with the DNN model ..." >> nn.log 

python3 $pdnndir/cmds/run_Extract_Feats.py --data "test.part*.blp" \
                                          --nnet-param dnn.param --nnet-cfg dnn.cfg \
                                          --output-file "dnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >> nn.log 2>&1

echo "Adam Test" >> testing.log
python3 $pdnndir/cmds/show_results.py dnn.classify.pickle.gz "test.part*.blp"  >> testing.log

