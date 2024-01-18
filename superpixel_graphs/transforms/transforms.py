# Author: Julia Pelayo Rodrigues 

from typing import Any, List
import torch
from torch_geometric.data import Data

from . import functional as F

__all__ = [
    "ToSuperpixelGraphGreyscale",
    "ToSuperpixelGraphColor",
    "NormalizeColor"
]

class ToSuperpixelGraphGreyscale(torch.nn.Module):
    """Transform the input greyscale image into a superpixel graph

    Args:
        n_segments (int): desired number of superpixels/nodes 
        segmentation_method (SegmentationMethod): desired segmentation method enum 
            defined by :class: `superpixel_graphs.transforms.SegmentationMethod`
        compactness (float): SLIC compactness parameter, only used when segmentation_method is
            `SegmentationMethod.SLIC`
        graph_type (GraphType): how the graph's neighborhood is defined
        features (List[FeatureGreyscale]): selected features, default is all available, 
            as defined in :class: `superpixel_graphs.transforms.FeatureGreyscale`

    """

    def __init__(self, 
                 n_segments: int = 75, 
                 segmentation_method: F.SegmentationMethod = F.SegmentationMethod.SLIC0, 
                 compactness: float = 0.1, 
                 graph_type: F.GraphType = F.GraphType.RAG, 
                 features: List[F.FeatureGreyscale] = None
    ) -> None:
        super().__init__()
        self.n_segments = n_segments
        self.segmentation_method = segmentation_method
        self.compactness = compactness
        self.graph_type = graph_type
        self.features = features
    
    def forward(self, img:Any) -> Data:
        """
        Args: 
            img(PIL Image or Tensor): Image to be transformed

        Returns:
            torch_geometric Data: the resulting graph
        """
        return F.to_superpixel_graph_greyscale(img, 
                                               n_segments=self.n_segments, 
                                               segmentation_method=self.segmentation_method,
                                               compactness=self.compactness, 
                                               graph_type=self.graph_type, 
                                               features=self.features)
    
    def __repr__(self) -> str:
        detail = f"(n_segments={self.n_segments}, segmentation_method={self.segmentation_method}, graph_type={self.graph_type})"
        return f"{self.__class__.__name__}{detail}"

class ToSuperpixelGraphColor(torch.nn.Module):
    """Transform the input RGB image into a superpixel graph

    Args:
        n_segments (int): desired number of superpixels/nodes 
        segmentation_method (SegmentationMethod): desired segmentation method enum 
            defined by :class: `superpixel_graphs.transforms.SegmentationMethod`
        compactness (float): SLIC compactness parameter, only used when segmentation_method is
            `SegmentationMethod.SLIC`
        graph_type (GraphType): how the graph's neighborhood is defined
        features (List[FeatureColor]): selected features, default is all available, 
            as defined in :class: `superpixel_graphs.transforms.FeatureColor`

    """
    def __init__(self, 
                 n_segments: int = 75, 
                 segmentation_method: F.SegmentationMethod = F.SegmentationMethod.SLIC0, 
                 compactness: float = 0.1, 
                 graph_type: F.GraphType = F.GraphType.RAG, 
                 features: List[F.FeatureColor] = None
    ) -> None:
        super().__init__()
        self.n_segments = n_segments
        self.segmentation_method = segmentation_method
        self.compactness = compactness
        self.graph_type = graph_type
        self.features = features
    
    def forward(self, img:Any) -> Data:
        """
        Args: 
            img(PIL Image or Tensor): Image to be transformed
            
        Returns:
            torch_geometric Data: the resulting graph
        """
        return F.to_superpixel_graph_color(img, 
                                           n_segments=self.n_segments, 
                                           segmentation_method=self.segmentation_method,
                                           compactness=self.compactness, 
                                           graph_type=self.graph_type, 
                                           features=self.features)
    
    def __repr__(self) -> str:
        detail = f"(n_segments={self.n_segments}, segmentation_method={self.segmentation_method}, graph_type={self.graph_type})"
        return f"{self.__class__.__name__}{detail}"

class NormalizeColor(torch.nn.Module):
    def __init__(self, 
                 mean: List[float], 
                 std: List[float]
    ):
        
        super().__init__()
        self.mean = mean 
        self.std  = std 
    
    def forward(self, graph:Data) -> Data:
        return F.normalize_color(graph, self.mean, self.std)