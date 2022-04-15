#!/bin/bash

### Create conda env and install dependencies:

conda create -n qagnn python=3.8
conda activate qagnn

conda install pytorch cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg
conda install -c conda-forge transformers
pip install matplotlib

### Download preprocessed data first:

wget https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip
unzip data_preprocessed_release.zip
mv data_preprocessed_release data


### Run the code

python main.py --lr 1e-5 --bs 64 --epochs 2 --pos-weight 4 --warmup-ratio 0.1 --sched=linear
# or
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1e-5 --bs 64 --epochs 3 --pos-weight 4 --warmup-ratio 0.1 --sched=linear