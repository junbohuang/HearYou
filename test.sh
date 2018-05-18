#!/bin/bash

# two variables you need to set
pdnndir=./packages/pdnn  # pointer to PDNN
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

# activate python3 environment
source activate python3
echo "python3 environment activated!"


# prepare testing Data
echo "Prepare data for testing"

python3 ./Conceptor/Emotion/PrepTestData.py ../../data/iemocap/wav_test

# switch to python2 environment.
echo "switching to python2 environment."
source deactivate
source activate python2

echo "Extracting test features with the DNN model ..."

python $pdnndir/cmds/run_Extract_Feats.py --data "./Conceptor/Emotion/test.pickle.gz" \
--nnet-param ./Conceptor/Emotion/dnn.param --nnet-cfg ./Conceptor/Emotion/dnn.cfg \
--output-file "./Conceptor/Emotion/testfeature.pickle.gz" --layer-index -1 \
--batch-size 100 >& ./Conceptor/Emotion/dnn.testing.log


# switch to python3 environment.
source deactivate
source activate python3

echo "Annotate results into Results.txt"

python3 ./Conceptor/Emotion/Annotate.py

echo "getting accuracy ..."

python3 accuracy.py

source deactivate 

echo "are you satisfied with the result?"
