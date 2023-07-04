# Author: Julia Pelayo Rodrigues 

import torch
import torchvision.datasets as datasets 
import torchvision.transforms as T
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import skimage as ski
import networkx as nx
import time

try:
    from superpixel_graphs.graph_builder import greyscale_features
except ImportError:
    extension_availabe = False
else:
    extension_availabe = True


# graph types 
graph_types_dict = {'RAG' : 0,
                    '1NNSpatial' : 1,
                    '2NNSpatial' : 2,
                    '4NNSpatial' : 3,
                    '8NNSpatial' : 4,
                    '16NNSpatial': 5,
                    '1NNFeature' : 6,
                    '2NNFeature' : 7,
                    '4NNFeature' : 8,
                    '8NNFeature' : 9,
                    '16NNFeature': 10 }

# slic methods
slic_methods_dict = {'SLIC0': 0,
                     'SLIC': 1,
                     'grid': 2 }

# features 
greyscale_features_dict = {'avg_color': 0,
                           'std_deviation_color': 1, 
                           'centroid': (2, 3), 
                           'std_deviation_centroid': (4, 5), 
                           'num_pixels': 6}

color_features_dict     = {'avg_color': (0,1,2),
                           'std_deviation_color': (3,4,5), 
                           'centroid': (6, 7), 
                           'std_deviation_centroid': (8, 9), 
                           'num_pixels': 10, 
                           'avg_color_hsv': (11, 12, 13), 
                           'std_deviation_hsv': (14, 15, 16)}

std_features = ['avg_color',
                'std_deviation_color',
                'centroid',
                'std_deviation_centroid']

# computing greyscale graphs 

def greyscale_graph(img, label=None, n_segments=75, segmentation_method='SLIC0', compactness=0.1, graph_type='RAG', selected_features=None):
    if extension_availabe:
        features, edge_index, segments = greyscale_features(img, 
                                                            n_segments, 
                                                            graph_type, 
                                                            segmentation_method,
                                                            compactness)
    else:
        features, edge_index, segments = greyscale_features_python(img, 
                                                                   n_segments, 
                                                                   graph_type, 
                                                                   segmentation_method,
                                                                   compactness)
    posi = greyscale_features_dict['centroid']
    pos = features[:, posi[0] : posi[1]+1]
    return Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), label)
