from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np 
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor
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
    """Transform the given greyscale image into a superpixel graph

    Args:
        img(PIL Image or Tensor): Image to be transformed
        n_segments (int): desired number of superpixels/nodes 
        segmentation_method (SegmentationMethod): desired segmentation method enum 
            defined by :class: `superpixel_graphs.transforms.SegmentationMethod`
        compactness (float): SLIC compactness parameter, only used when segmentation_method is
            `SegmentationMethod.SLIC`
        graph_type (GraphType): how the graph's neighborhood is defined
        features (List[Feature]): selected features, default is all available, 
            as defined in :class: `superpixel_graphs.transforms.FeatureGreyscale`

    Returns:
        torch_geometric Data: the resulting graph
    """
    if type(img) == Image.Image:
        img = to_tensor(img)
    _, dim0, dim1 = img.shape
    img_np = img.view(dim0, dim1).numpy() 
    x, edge_index, _= greyscale_features(img_np, 
                                         n_segments, 
                                         graph_type.value, 
                                         segmentation_method.value,
                                         compactness)
    pos = features[:, FeatureGreyscale.CENTROID.value]
    if features:
        feature_mask = []
        [feature_mask.extend(feature.value if isinstance(feature.value, list) else [feature.value]) for feature in features]
        return Data(x=torch.from_numpy(x[:,feature_mask]).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float))
    return Data(x=torch.from_numpy(x).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float))

def to_segments_greyscale(
        img: Any, 
        n_segments: int = 75, 
        segmentation_method: SegmentationMethod = SegmentationMethod.SLIC0, 
        compactness: float = 0.1, 
) -> Any:
    """Segmetns the given greyscale image into superpixels 

    Args:
        img(PIL Image or Tensor): Image to be transformed
        n_segments (int): desired number of superpixels/nodes 
        segmentation_method (SegmentationMethod): desired segmentation method enum 
            defined by :class: `superpixel_graphs.transforms.SegmentationMethod`
        compactness (float): SLIC compactness parameter, only used when segmentation_method is
            `SegmentationMethod.SLIC`

    Returns:
        Array: integer mask indicting superpixel labels 
    """
    if type(img) == Image.Image:
        img = to_tensor(img) # to_tensor transforms from PIL img [0,255], (H x W x C) to FloatTensor [0.0,1.0], (C x H x W)
    _, dim0, dim1 = img.shape
    img_np = img.view(dim0, dim1).numpy() 
    _, _, segments = greyscale_features(img_np, 
                                        n_segments, 
                                        GraphType.RAG.value, 
                                        segmentation_method.value,
                                        compactness)
    return segments


def to_superpixel_graph_color(
        img: Any, 
        n_segments: int = 75, 
        segmentation_method: SegmentationMethod = SegmentationMethod.SLIC0, 
        compactness: float = 0.1, 
        graph_type: GraphType = GraphType.RAG, 
        features: List[FeatureColor] = None
) -> Data:
    """Transform the given RGB image into a superpixel graph

    Args:
        img(PIL Image or Tensor): Image to be transformed
        n_segments (int): desired number of superpixels/nodes 
        segmentation_method (SegmentationMethod): desired segmentation method enum 
            defined by :class: `superpixel_graphs.transforms.SegmentationMethod`
        compactness (float): SLIC compactness parameter, only used when segmentation_method is
            `SegmentationMethod.SLIC`
        graph_type (GraphType): how the graph's neighborhood is defined
        features (List[FeatureColor]): selected features, default is all available, 
            as defined in :class: `superpixel_graphs.transforms.FeatureColor`
            
    Returns:
        torch_geometric Data: the resulting graph
    """
    if type(img) == Image.Image:
        img = to_tensor(img)
    img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
    x, edge_index, _= color_features(img_np, 
                                            n_segments, 
                                            graph_type.value, 
                                            segmentation_method.value,
                                            compactness)
    pos = x[:, FeatureColor.CENTROID.value]
    if features:
        feature_mask = []
        [feature_mask.extend(feature.value if isinstance(feature.value, list) else [feature.value]) for feature in features]
        return Data(x=torch.from_numpy(x[:,feature_mask]).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float))
    else:
        return Data(x=torch.from_numpy(x).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float))

def to_segments_greyscale(
        img: Any, 
        n_segments: int = 75, 
        segmentation_method: SegmentationMethod = SegmentationMethod.SLIC0, 
        compactness: float = 0.1, 
        graph_type: GraphType = GraphType.RAG 
):
    if type(img) == Image.Image:
        img = to_tensor(img)
    _, dim0, dim1 = img.shape
    img_np = img.view(dim0, dim1).numpy() 
    _, _, segments = greyscale_features(img_np, 
                                        n_segments, 
                                        graph_type.value, 
                                        segmentation_method.value,
                                        compactness)
    return img_np, segments

def to_segments_color(
        img: Any, 
        n_segments: int = 75, 
        segmentation_method: SegmentationMethod = SegmentationMethod.SLIC0, 
        compactness: float = 0.1, 
        graph_type: GraphType = GraphType.RAG 
):
    if type(img) == Image.Image:
        img = to_tensor(img)
    img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
    _, _, segments = color_features(img_np, 
                                    n_segments, 
                                    graph_type.value, 
                                    segmentation_method.value,
                                    compactness)
    return img_np, segments