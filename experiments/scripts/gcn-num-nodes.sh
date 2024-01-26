#!/bin/sh
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-num-nodes --n_segments 10
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-num-nodes --n_segments 20
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-num-nodes --n_segments 50
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-num-nodes --n_segments 100
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-num-nodes --n_segments 200
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-num-nodes --n_segments 400

python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-num-nodes --n_segments 10
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-num-nodes --n_segments 20
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-num-nodes --n_segments 50
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-num-nodes --n_segments 100
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-num-nodes --n_segments 200
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-num-nodes --n_segments 400

python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-num-nodes --n_segments 10
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-num-nodes --n_segments 20
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-num-nodes --n_segments 50
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-num-nodes --n_segments 100
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-num-nodes --n_segments 200
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-num-nodes --n_segments 400

python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-num-nodes --n_segments 10
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-num-nodes --n_segments 20
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-num-nodes --n_segments 50
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-num-nodes --n_segments 100
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-num-nodes --n_segments 200
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-num-nodes --n_segments 400

python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-num-nodes --n_segments 10
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-num-nodes --n_segments 20
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-num-nodes --n_segments 50
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-num-nodes --n_segments 100
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-num-nodes --n_segments 200
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-num-nodes --n_segments 400

