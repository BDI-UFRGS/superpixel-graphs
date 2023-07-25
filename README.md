# Superpixel-Graphs library 

Parameterizable graph-building from images for [pytorch](https://pytorch.org/) with [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html#).

## Overview 

The purpose of this package is to provide a fast and customizable way to build graphs from images over-segmented into superpixel so that they can be used as input for Graph Neural Networks in a highly parameterizable manner. 

It comprises the following components:
- transforms: containing torchvision like composable transforms 
- datasets: as optimized as possible adaptations of torchvision datasets into PyG datasets

When building graphs, it is possible to choose:

- **Segmentation method**: we currently support SLIC superpixel segmentation from Achanta et al.: [SLIC Superpixels Compared to State-of-the-Art Superpixel Methods](https://ieeexplore.ieee.org/abstract/document/6205760). In it's original (SLIC) and zero-parameter variants (SLIC0). The number of desired superpixels and compactness (smothness factor, when using the original SLIC) can be configured.
- **Graph edge-building methods**: when building a graph from a superpixel segmented image, each superpixel is a node. The edges between nodes can be built by connecting adjacent regions (RAG) or connecting the K Nearest-Neighboors using spatial distance or a combined spatial (SPATIAL_KNN) a color distance (FEATURE_KNN). 
- **Features**: we support the computing of the following features for each node/superpixel using it's color and geometric information:
    - Average color
    - Color standard deviation 
    - Average color in HSV space
    - HSV color standard deviation 
    - Geometric centroid
    - Standard deviation from centroid 
    - Number of pixels

### Transforms examples

For greyscale datasets

```python
import torch
import torchvision 
import torchvision.datasets as datasets

import superpixel_graphs.transforms as T

ds = datasets.MNIST(root='mnist/', download=True, transform=T.ToSuperpixelGraphGreyscale())
```

For color datasets

```python
import torch
import torchvision 
import torchvision.datasets as datasets

import superpixel_graphs.transforms as T

ds = datasets.CIFAR10(root='cifar/', transform=T.ToSuperpixelGraphColor())
```

Transforms are also available in functional forms in ```superpixel_graphs.transform.functional``` as ```to_superpixel_graph_greyscale``` and ```to_superpixel_graph_color```.

### Datasets examples 

```python
from superpixel_graphs.datasets import SuperPixelGraphMNIST

ds = SuperPixelGraphMNIST(root=None, 
                          train=False,
                          n_segments=75,
                          compactness=0.1,
                          graph_type='RAG',
                          slic_method='SLIC')

```

Currently suported datasets are:
- SuperPixelGraphMNIST
- SuperPixelGraphFashionMNIST
- SuperPixelGraphCIFAR10
- SuperPixelGraphCIFAR100
- SuperPixelGraphSTL10

Also available are two base classes that can be extended to create new superpixel datasets
- ColorSLIC
- GreyscaleSLIC

## Installation

This package is available on Linux. You can install it with pip:

```
pip install superpixel_graphs@git+https://github.com/BDI-UFRGS/superpixel-graphs.git
```

### Requirements

This package uses a OpenCV C++ extension for image segmentation and feature computation. 
Here is how to [install OpenCV for Linux](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

Naturally, [pytorch and torchvision](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html#) are also required.



