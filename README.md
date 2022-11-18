# FedCAMS

This repository contains the PyTorch implementation of Federated AMSGrad with Max Stabilization (FedAMS), and Federated Communication compressed AMSGrad with Max Stabilization (FedCAMS) in <https://arxiv.org/pdf/2205.02719.pdf> (accepted by ICML 2022).

## Prerequisites
Pytorch 1.11.0

CUDA 11.3

## Running the experiments

To run the experiment for FedAMS:

```
python3 federated_main.py --model=resnet --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedams --local_lr=0.01 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3
```
To run the experiment for FedCAMS:
```
python3 federated_main-ef.py --model=resnet --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedams --local_lr=0.01 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3
```
## Options
The default values for various paramters parsed to the experiment are given in ```options.py```.

```--dataset:``` Default: 'cifar10'. Options: 'mnist', 'fmnist', 'cifar100'.

```--model:``` Default: 'cnn'. Options: 'mlp', 'resnet', 'convmixer'.

```--gpu:``` To use cuda, set to a specific GPU ID.

```--epochs:``` Number of rounds of training.

```--local_ep:``` Number of local epochs.

```--local_lr:``` Learning rate for local update.

```--lr:``` Learning rate for global update.

```--local_bs:``` Local update batch size.

```--iid:``` Default set to IID. Set to 0 for non-IID.

```--num_users:``` Number of users. Default is 100.

```--frac:``` Fraction of users to be used for federated updates. Default is 0.1.

```--optimizer:``` Default: 'fedavg'. Options: 'fedadam', 'fedams'.

```--compressor:``` Compression strategy. Default: 'sign'. Options: 'topk64', 'topk128', 'topk256'.

## Citation
Please check our paper for technical details and full results.
```
@inproceedings{wang2022communication,
  title={Communication-Efficient Adaptive Federated Learning},
  author={Wang, Yujia and Lin, Lu and Chen, Jinghui},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2022}
}
 
```

