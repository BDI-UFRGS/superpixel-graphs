#!/bin/sh 

python3 train-model.py -m EfficientNet --patience 5 -lr 0.001 --dataset mnist -f EfficientNet/mnist 
python3 train-model.py -m EfficientNet --patience 5 -lr 0.001 --dataset fashion_mnist -f EfficientNet/fashion_mnist 
python3 train-model.py -m EfficientNet --patience 5 -lr 0.001 --dataset cifar10 -f EfficientNet/cifar10 
python3 train-model.py -m EfficientNet --patience 5 -lr 0.001 --dataset cifar100 -f EfficientNet/cifar100 
python3 train-model.py -m EfficientNet --patience 5 -lr 0.001 --dataset stl10 -f EfficientNet/stl10 
