#!/bin/bash

# setup virtualenv
ENV_PATH=$HOME/candy
virtualenv $ENV_PATH
source $ENV_PATH/bin/activate

# download all dependencies
pip install -r requirements.txt

# get data
./get_data.sh

# train with output, takes less then 2 minutes using GPU
bsub -o result.log -R 'rusage[ngpus_excl_p=1, mem=4096]' 'python model.py --pretrained vec/fasttext.vec --prediction submission.csv'