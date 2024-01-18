import torch

from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torchvision.transforms import v2

from torch_geometric.loader import DataLoader
import superpixel_graphs.transforms as GT
from superpixel_graphs.transforms import ToSuperpixelGraphColor

import time

def test(data_loader):
    for X, Y in data_loader:
        X.to('cuda')
        Y.to('cuda')
        X = None 
        Y = None

if __name__ == '__main__':
    iterations = 1000

    preprocess = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_augmentation = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5)
    # v2.RandomCrop(32, padding=4)
    ])

    graph_transform = ToSuperpixelGraphColor(n_segments=75, # approximate number of nodes in the graph
                                        graph_type=GT.GraphType.RAG, # how neighborhoods are built: region adjacency graphs
                                        features=[GT.FeatureColor.AVG_COLOR,         # list of features that describe each node
                                                GT.FeatureColor.CENTROID,
                                                GT.FeatureColor.STD_DEV_COLOR,
                                                GT.FeatureColor.STD_DEV_CENTROID])

    transform = T.Compose([preprocess, 
                        data_augmentation])#, 
                        # graph_transform])

    # ds_train = CIFAR10(root='./cifar10/train', train=True, transform=transform, download=True)
    ds_test  = CIFAR10(root='./cifar10/test', train=False, transform=transform, download=True)

    test_loader = DataLoader(ds_test, batch_size=128)
    # train_loader = DataLoader(ds_train, batch_size=128)

    for i in range(iterations):
        print(f'Iteration {i}')
        t = time.time()
        test(test_loader)
        t = time.time() - t
        print(f' End of iteration {i} ({t}s)')    
