# Reproducibility of QAGNN Paper

## Paper Summary

The paper titled [QAGNN: A Graph Neural Network Model for Question Answering with Graphs](https://arxiv.org/abs/2104.06378) presents a novel approach for question answering tasks using graph neural networks. The proposed QAGNN model leverages graph-based structures to enhance the understanding and representation of the question and answer pairs. The approach aims to improve the performance of question answering systems by incorporating external knowledge graphs, which helps in better capturing the relationships and dependencies between different entities in the data.

<p align="center">
  <img src="https://github.com/michiyasunaga/qagnn/raw/main/figs/overview.png" width="1000" title="Overview of QA-GNN" alt="">
</p>

## Running Instructions

Follow the instructions below to set up the environment, download the necessary data, and run the code.

### 1. Create Conda Environment and Install Dependencies

```bash
# Create conda environment
conda create -n qagnn python=3.8
conda activate qagnn

# Install dependencies
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg
conda install -c conda-forge transformers
pip install matplotlib
```

### 2. Download Preprocessed Data
```bash
# Download and unzip data
wget https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip
unzip data_preprocessed_release.zip
mv data_preprocessed_release data
```

### 3. Run the Code
```bash
# Run the main script to train and evaluate the model
CUDA_VISIBLE_DEVICES=0 python main.py --lr 1e-5 --bs 64 --epochs 3 --pos-weight 4 --warmup-ratio 0.1 --sched=linear
```