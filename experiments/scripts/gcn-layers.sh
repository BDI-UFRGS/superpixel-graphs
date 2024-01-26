#!/bin/sh
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-layers --n_layers 1
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-layers --n_layers 2
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-layers --n_layers 3
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-layers --n_layers 4
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-layers --n_layers 5

python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-layers --n_layers 1
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-layers --n_layers 2
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-layers --n_layers 3
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-layers --n_layers 4
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-layers --n_layers 5

python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-layers --n_layers 1
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-layers --n_layers 2
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-layers --n_layers 3
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-layers --n_layers 4
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-layers --n_layers 5

python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-layers --n_layers 1
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-layers --n_layers 2
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-layers --n_layers 3
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-layers --n_layers 4
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-layers --n_layers 5

python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-layers --n_layers 1
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-layers --n_layers 2
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-layers --n_layers 3
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-layers --n_layers 4
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-layers --n_layers 5
