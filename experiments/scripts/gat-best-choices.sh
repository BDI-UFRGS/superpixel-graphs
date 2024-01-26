#!/bin/sh 

python3 train-model.py -m GAT --dataset mnist -f GAT/mnist-best-choices --n_segments 50 
python3 train-model.py -m GAT --dataset fashion_mnist -f GAT/fashion_mnist-best-choices --n_segments 50  
python3 train-model.py -m GAT --dataset cifar10 -f GAT/cifar10-best-choices --n_segments 400 
python3 train-model.py -m GAT --dataset cifar100 -f GAT/cifar100-best-choices --n_segments 200 
python3 train-model.py -m GAT --dataset stl10 -f GAT/stl10-best-choices --features "avg_color centroid std_deviation_color std_deviation_centroid avg_color_hsv" --n_segments 400 
