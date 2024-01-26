#!/bin/sh
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-layers --n_layers 1
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-layers --n_layers 2
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-layers --n_layers 3
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-layers --n_layers 4
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-layers --n_layers 5

python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-layers --n_layers 1
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-layers --n_layers 2
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-layers --n_layers 3
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-layers --n_layers 4
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-layers --n_layers 5

python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-layers --n_layers 1
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-layers --n_layers 2
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-layers --n_layers 3
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-layers --n_layers 4
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-layers --n_layers 5

python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-layers --n_layers 1
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-layers --n_layers 2
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-layers --n_layers 3
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-layers --n_layers 4
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-layers --n_layers 5

python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-layers --n_layers 1
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-layers --n_layers 2
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-layers --n_layers 3
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-layers --n_layers 4
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-layers --n_layers 5
