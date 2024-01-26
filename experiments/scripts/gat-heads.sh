#!/bin/sh
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-heads --n_heads 1
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-heads --n_heads 2
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-heads --n_heads 4
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-heads --n_heads 8
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-heads --n_heads 16

python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-heads --n_heads 1
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-heads --n_heads 2
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-heads --n_heads 4
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-heads --n_heads 8
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-heads --n_heads 16

python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-heads --n_heads 1
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-heads --n_heads 2
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-heads --n_heads 4
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-heads --n_heads 8
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-heads --n_heads 16

python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-heads --n_heads 1
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-heads --n_heads 2
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-heads --n_heads 4
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-heads --n_heads 8
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-heads --n_heads 16

python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-heads --n_heads 1
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-heads --n_heads 2
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-heads --n_heads 4
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-heads --n_heads 8
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-heads --n_heads 16

