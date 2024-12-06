#!/bin/bash
echo "Hello OSPool from Job $1 running on `hostname` with Subset $2"

pip install transformers datasets

# Run the PyTorch model on this subset
python xlm.py --save-model --epochs 5
