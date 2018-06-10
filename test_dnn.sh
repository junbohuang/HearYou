#!/usr/bin/env bash

source activate python2

# two variables you need to set
pdnndir=./packages/pdnn  # pointer to PDNN
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32, exception_verbosity=high

# train DNN model
echo "Training the DNN model ..."

python2 $pdnndir/cmds/run_DNN.py --train-data "./Conceptor/Emotion/train.pickle.gz" \
--valid-data "./Conceptor/Emotion/valid.pickle.gz" \
--nnet-spec "325:500:5" --wdir ./ \
--l2-reg 0.0001 --lrate "C:0.2:10" --model-save-step 5 \
--param-output-file ./Conceptor/Emotion/dnn.param --cfg-output-file ./Conceptor/Emotion/dnn.cfg  >& ./Conceptor/Emotion/dnn.training.log

source deactivate