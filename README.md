# mutationTCN
This is the code for the paper Prediction of Mutation Effects using a Deep Temporal Convolutional Network (paper in progress).

The code is compatible with tensorflow-gpu=1.10.0 and python=2.7.

## Example code run:
1) Unsupervised model
``` bash
CUDA_VISIBLE_DEVICES=0 python train.py ../data/
```
2) Semi-supervised model
``` bash
CUDA_VISIBLE_DEVICES=0 python train_ssTCN_pretraining.py ../data/

CUDA_VISIBLE_DEVICES=0 python train_ssTCN_training.py ../data/
```
