# mutationTCN
This is the code for the paper Prediction of Mutation Effects using a Deep Temporal Convolutional Network.
https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btz873/5634146

The code is compatible with tensorflow-gpu=1.10.0 and python=2.7.

## Example code run:
1) Unsupervised model
``` bash
cd unsupervised/

CUDA_VISIBLE_DEVICES=0 python train.py ../data/
```
2) Semi-supervised model
``` bash
cd semisupervised/

CUDA_VISIBLE_DEVICES=0 python train_ssTCN_pretraining.py ../data/

CUDA_VISIBLE_DEVICES=0 python train_ssTCN_training.py ../data/
```
