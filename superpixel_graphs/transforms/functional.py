from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np 
import torch
from PIL import Image
from torch import Tensor
from torch_geometric.data import Data
from skimage.segmentation import slic

from .ext import greyscale_features, color_features

class GraphType(Enum):
    RAG = 0
    SPATIAL_1NN  = 1
    SPATIAL_2NN  = 2
    SPATIAL_4NN  = 3
    SPATIAL_8NN  = 4
    SPATIAL_16NN = 5
    FEATURE_1NN  = 6
    FEATURE_2NN  = 7
    FEATURE_4NN  = 8
    FEATURE_8NN  = 9
    FEATURE_16NN = 10 

class SegmentationMethod(Enum):
    SLIC0 = 0
    SLIC  = 1
    GRID  = 2

class FeatureGreyscale(Enum):
    AVG_COLOR = 0
    STD_DEV_COLOR = 1
    CENTROID = [2,3]
    STD_DEV_CENTROID = [4,5]
    NUM_PIXELS = 6

class FeatureColor(Enum):
    AVG_COLOR = [0,1,2]
    STD_DEV_COLOR = [3,4,5]
    CENTROID = [6,7]
    STD_DEV_CENTROID = [8,9]
    NUM_PIXELS = 10
    AVG_COLOR_HSV = [11,12,13]
    STD_DEV_HSV = [14,15,16]


def to_superpixel_graph_greyscale(
        img: Any, 
        n_segments: int = 75, 
        segmentation_method: SegmentationMethod = SegmentationMethod.SLIC0, 
        compactness: float = 0.1, 
        graph_type: GraphType = GraphType.RAG, 
        features: List[FeatureGreyscale] = None
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
    features, edge_index, _= greyscale_features(img, 
                                                n_segments, 
                                                graph_type.value, 
                                                segmentation_method.value,
                                                compactness)
    posi = features_mapping_greyscale[Feature.CENTROID]
    pos = features[:, posi[0] : posi[1]+1]
    return Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), y=label)
