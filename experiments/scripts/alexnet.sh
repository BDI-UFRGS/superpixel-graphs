#!/bin/sh 

python3 train-model.py -m AlexNet -lr 0.00001 --dataset mnist -f AlexNet/mnist 
python3 train-model.py -m AlexNet -lr 0.00001 --dataset fashion_mnist -f AlexNet/fashion_mnist 
python3 train-model.py -m AlexNet -lr 0.00001 --dataset cifar10 -f AlexNet/cifar10 
python3 train-model.py -m AlexNet -lr 0.00001 --dataset cifar100 -f AlexNet/cifar100 
python3 train-model.py -m AlexNet -lr 0.00001 --dataset stl10 -f AlexNet/stl10 
