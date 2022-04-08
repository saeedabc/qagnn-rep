#!/bin/bash

### Create conda env and install dependencies:

conda create -n qagnn python=3.8
conda activate qagnn

conda install pytorch cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg
conda install -c conda-forge transformers


### Download preprocessed data first:

wget https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip
unzip data_preprocessed_release.zip
mv data_preprocessed_release data


### Run the code

python main.py
# or
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py