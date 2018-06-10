#!/bin/bash

# two variables you need to set
pdnndir=./packages/pdnn  # pointer to PDNN
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32, exception_verbosity=high


source activate python2

echo "Extracting Features for ELM training ..."

python $pdnndir/cmds/run_Extract_Feats.py --data "./Conceptor/Emotion/ELMtrain.pickle.gz" \
--nnet-param ./Conceptor/Emotion/dnn.param --nnet-cfg ./Conceptor/Emotion/dnn.cfg \
--output-file "./Conceptor/Emotion/ELMfeature.pickle.gz" --layer-index -1 \
--batch-size 100 >& ./Conceptor/Emotion/ELMdnn.testing.log

# switch to python3 environment.
source deactivate
echo "model trained. please run test.sh."



