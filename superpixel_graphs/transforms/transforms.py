# Author: Julia Pelayo Rodrigues 

from typing import Any, List
import torch
from torch_geometric.data import Data

from . import functional as F

class ToSuperpixelGraphGreyscale(torch.nn.Module):
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
        return F.to_superpixel_graph_color(img, 
                                           n_segments=self.n_segments, 
                                           segmentation_method=self.segmentation_method,
                                           compactness=self.compactness, 
                                           graph_type=self.graph_type, 
                                           features=self.features)
    
    def __repr__(self) -> str:
        detail = f"(n_segments={self.n_segments}, segmentation_method={self.segmentation_method}, graph_type={self.graph_type})"
        return f"{self.__class__.__name__}{detail}"
