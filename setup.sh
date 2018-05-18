#!/bin/bash

# activate python3 environment
source activate python3
echo "python3 environment activated!"

echo "Setting up..."

python3 basic_setup.py

source deactivate 

echo "basic environment set up! please run train.sh."
