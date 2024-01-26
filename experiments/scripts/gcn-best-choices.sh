#!/bin/sh 

python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-best-choices                 --n_segments  50 --graph_type RAG --features "avg_color centroid num_pixels std_deviation_centroid std_deviation_color"
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-best-choices --n_segments 200 --graph_type 1NNFeature --features "avg_color centroid num_pixels std_deviation_centroid std_deviation_color"
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-best-choices             --n_segments 400 --graph_type 1NNFeature --features "avg_color avg_color_hsv centroid num_pixels std_deviation_centroid std_deviation_color std_deviation_color_hsv"
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-best-choices           --n_segments 200 --graph_type 1NNFeature --features "avg_color avg_color_hsv centroid num_pixels std_deviation_centroid std_deviation_color std_deviation_color_hsv"
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-best-choices                 --n_segments 400 --graph_type 2NNFeature --features "avg_color centroid num_pixels std_deviation_color std_deviation_centroid avg_color_hsv" 
