#!/usr/bin/env bash

# activate python3 environment
source activate python3
echo "python3 environment activated!"

# two variables you need to set
pdnndir=./packages/pdnn  # pointer to PDNN
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# prepare Training Data
echo "Prepare data for training"

python3 ./Conceptor/Emotion/PrepTrainData.py ./data/iemocap/wav_train ./data/iemocap/wav_valid

# switch to python2 environment.
source deactivate