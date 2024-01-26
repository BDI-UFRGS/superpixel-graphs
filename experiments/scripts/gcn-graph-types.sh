#!/bin/sh 

python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type RAG 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 1NNSpatial 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 2NNSpatial 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 4NNSpatial 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 8NNSpatial 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 16NNSpatial 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 1NNFeature 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 2NNFeature 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 4NNFeature 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 8NNFeature 
python3 train-model.py -m GCN --dataset mnist -f GCN/mnist-graph-types --graph_type 16NNFeature 

python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type RAG 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 1NNSpatial 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 2NNSpatial 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 4NNSpatial 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 8NNSpatial 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 16NNSpatial 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 1NNFeature 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 2NNFeature 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 4NNFeature 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 8NNFeature 
python3 train-model.py -m GCN --dataset fashion_mnist -f GCN/fashion_mnist-graph-types --graph_type 16NNFeature 

python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type RAG 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 1NNSpatial 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 2NNSpatial 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 4NNSpatial 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 8NNSpatial 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 16NNSpatial 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 1NNFeature 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 2NNFeature 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 4NNFeature 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 8NNFeature 
python3 train-model.py -m GCN --dataset cifar10 -f GCN/cifar10-graph-types --graph_type 16NNFeature 

python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type RAG 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 1NNSpatial 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 2NNSpatial 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 4NNSpatial 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 8NNSpatial 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 16NNSpatial 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 1NNFeature 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 2NNFeature 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 4NNFeature 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 8NNFeature 
python3 train-model.py -m GCN --dataset cifar100 -f GCN/cifar100-graph-types --graph_type 16NNFeature 

python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type RAG 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 1NNSpatial 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 2NNSpatial 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 4NNSpatial 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 8NNSpatial 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 16NNSpatial 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 1NNFeature 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 2NNFeature 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 4NNFeature 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 8NNFeature 
python3 train-model.py -m GCN --dataset stl10 -f GCN/stl10-graph-types --graph_type 16NNFeature 
