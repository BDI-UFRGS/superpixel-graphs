#!/bin/sh
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-nodes --n_segments 10
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-nodes --n_segments 20
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-nodes --n_segments 50
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-nodes --n_segments 100
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-nodes --n_segments 200
python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-num-nodes --n_segments 400

python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-nodes --n_segments 10
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-nodes --n_segments 20
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-nodes --n_segments 50
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-nodes --n_segments 100
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-nodes --n_segments 200
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-num-nodes --n_segments 400

python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-nodes --n_segments 10
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-nodes --n_segments 20
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-nodes --n_segments 50
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-nodes --n_segments 100
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-nodes --n_segments 200
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-num-nodes --n_segments 400

python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-nodes --n_segments 10
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-nodes --n_segments 20
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-nodes --n_segments 50
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-nodes --n_segments 100
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-nodes --n_segments 200
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-num-nodes --n_segments 400

python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-nodes --n_segments 10
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-nodes --n_segments 20
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-nodes --n_segments 50
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-nodes --n_segments 100
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-nodes --n_segments 200
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-num-nodes --n_segments 400

