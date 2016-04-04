#!/bin/bash

# two variables you need to set
pdnndir=/data/ASR5/babel/ymiao/tools/pdnn  # pointer to PDNN
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# download mnist dataset
wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz	

# split the dataset to training, validation and testing sets
# you will see train.pickle.gz, valid.pickle.gz, test.pickle.gz
echo "Preparing datasets ..." > nn.log
python3 data_prep.py >> nn.log

# train CNN model
echo "Training the CNN model ..." >> nn.log
python3 $pdnndir/cmds/run_CNN.py --train-data "train.blp" \
                                --valid-data "valid.blp" \
                                --conv-nnet-spec "1x28x28:20,5x5,p2x2:50,5x5,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:200" --model-save-step 20 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >> nn.log

echo "Classifying with the CNN model ..."
python3 $pdnndir/cmds/run_Extract_Feats.py --data "test.blp" \
                                          --nnet-param cnn.param --nnet-cfg cnn.cfg \
                                          --output-file "cnn.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >> nn.log

python3 show_results.py cnn.classify.pickle.gz >> nn.log
