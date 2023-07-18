from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np 
import torch
from PIL import Image
from torch import Tensor
from torch_geometric.data import Data
from skimage.segmentation import slic

try:
    from .ext import greyscale_features, color_featres
except ImportError:
    extension_availabe = False
else:
    extension_availabe = True

class GraphType(Enum):
    RAG = 'RAG'
    SPATIAL_1NN  = '1NNSpatial'
    SPATIAL_2NN  = '2NNSpatial'
    SPATIAL_4NN  = '4NNSpatial'
    SPATIAL_8NN  = '8NNSpatial'
    SPATIAL_16NN = '16NNSpatial'
    FEATURE_1NN  = '1NNFeature'
    FEATURE_2NN  = '2NNFeature'
    FEATURE_4NN  = '4NNFeature'
    FEATURE_8NN  = '8NNFeature'
    FEATURE_16NN = '16NNFeature' 

graph_types_mapping = {
    GraphType.RAG : 0,
    GraphType.SPATIAL_1NN : 1,
    GraphType.SPATIAL_2NN : 2,
    GraphType.SPATIAL_4NN : 3,
    GraphType.SPATIAL_8NN : 4,
    GraphType.SPATIAL_16NN: 5,
    GraphType.FEATURE_1NN : 6,
    GraphType.FEATURE_2NN : 7,
    GraphType.FEATURE_4NN : 8,
    GraphType.FEATURE_8NN : 9,
    GraphType.FEATURE_16NN: 10 
}

class SegmentationMethod(Enum):
    SLIC0 = 'SLIC0'
    SLIC  = 'SLIC'
    GRID  = 'grid'

segmentation_method_mapping = {
    SegmentationMethod.SLIC0 : 0,
    SegmentationMethod.SLIC  : 1,
    SegmentationMethod.GRID  : 2
}

class Feature(Enum):
    AVG_COLOR = 'avg_color'
    STD_DEV_COLOR = 'std_deviation_color'
    CENTROID = 'centroid'
    STD_DEV_CENTROID = 'std_deviation_centroid'
    NUM_PIXELS = 'num_pixels'
    AVG_COLOR_HSV = 'avg_color_hsv'
    STD_DEV_HSV = 'std_deviation_hsv'

features_mapping_greyscale = {
    Feature.AVG_COLOR: 0,
    Feature.STD_DEV_COLOR: 1, 
    Feature.CENTROID: (2, 3), 
    Feature.STD_DEV_CENTROID: (4, 5), 
    Feature.NUM_PIXELS: 6
}

features_mapping_color = {
    Feature.AVG_COLOR: (0,1,2),
    Feature.STD_DEV_COLOR: (3,4,5), 
    Feature.CENTROID: (6, 7), 
    Feature.STD_DEV_CENTROID: (8, 9), 
    Feature.NUM_PIXELS: 10, 
    Feature.AVG_COLOR_HSV: (11, 12, 13), 
    Feature.STD_DEV_HSV: (14, 15, 16)
}

std_features = ['avg_color',
                'std_deviation_color',
                'centroid',
                'std_deviation_centroid']

def to_superpixel_graph_greyscale(
        img: Any, 
        n_segments: int = 75, 
        segmentation_method: SegmentationMethod = SegmentationMethod.SLIC0, 
        compactness: float = 0.1, 
        graph_type: GraphType = GraphType.RAG, 
        features: List[Feature] = None
) -> Data:
    """Transform the given image into a superpixel graph

    Args:
        img(PIL Image or Tensor): Image to be transformed
        n_segments (int): desired number of superpixels/nodes 
        segmentation_method (SegmentationMethod): desired segmentation method enum 
            defined by :class: `superpixel_graphs.transforms.SegmentationMethod`
        compactness (float): SLIC compactness parameter, only used when segmentation_method is
            `SegmentationMethod.SLIC`
        graph_type (GraphType): how the graph's neighborhood is defined
        features (List[Feature]): selected features, default is all available, 
            as defined in :class: `superpixel_graphs.transforms.Feature`

    """
    
    # make sure image is a tensor  
    if extension_availabe:
        features, edge_index, segments = greyscale_features(img, 
                                                            n_segments, 
                                                            graph_type, 
                                                            segmentation_method,
                                                            compactness)
    else:
        features, edge_index, segments = _greyscale_features(img, 
                                                             n_segments, 
                                                             graph_type, 
                                                             segmentation_method,
                                                             compactness)
    posi = features_mapping_greyscale[Feature.CENTROID]
    pos = features[:, posi[0] : posi[1]+1]
    return Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), y=label)

def _greyscale_features(
        img: Any,
        n_segments: int, 
        graph_type: GraphType, 
        sementation_method: SegmentationMethod, 
        compactness: float
):
    return None