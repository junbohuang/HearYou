#!/usr/bin/env bash

source activate python3

echo "Training the ELM..."

python3 ./Conceptor/Emotion/ELM_training.py

source deactivate