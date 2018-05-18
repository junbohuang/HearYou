#!/bin/bash

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
source activate python2

# train DNN model
echo "Training the DNN model ..."

python2 $pdnndir/cmds/run_DNN.py --train-data "./Conceptor/Emotion/train.pickle.gz" \
--valid-data "./Conceptor/Emotion/valid.pickle.gz" \
--nnet-spec "325:500:5" --wdir ./ \
--l2-reg 0.0001 --lrate "C:0.2:10" --model-save-step 5 \
--param-output-file ./Conceptor/Emotion/dnn.param --cfg-output-file ./Conceptor/Emotion/dnn.cfg  >& ./Conceptor/Emotion/dnn.training.log


echo "Extracting Features for ELM training ..."

python $pdnndir/cmds/run_Extract_Feats.py --data "./Conceptor/Emotion/ELMtrain.pickle.gz" \
--nnet-param ./Conceptor/Emotion/dnn.param --nnet-cfg ./Conceptor/Emotion/dnn.cfg \
--output-file "./Conceptor/Emotion/ELMfeature.pickle.gz" --layer-index -1 \
--batch-size 100 >& ./Conceptor/Emotion/ELMdnn.testing.log

# switch to python3 environment.
source deactivate
source activate python3

echo "Training the ELM..."

python3 ./Conceptor/Emotion/ELM_training.py

source deactivate
echo "model trained. please run test.sh."
