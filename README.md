# mutationTCN
This is the code for the paper Prediction of Mutation Effects using a Deep Temporal Convolutional Network (paper in progress).

The code is compatible with tensorflow-gpu 1.10.0 and python 2.7.

Example code run:
cd unsupervised/
CUDA_VISIBLE_DEVICES=# python train.py ../data/

cd semisupervised/
CUDA_VISIBLE_DEVICES=# python train_ssTCN_pretraining.py ../data/
CUDA_VISIBLE_DEVICES=# python train_ssTCN_training.py ../data/
