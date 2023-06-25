import mnist_slic 
import fashion_mnist_slic
import cifar10_slic
import cifar100_slic
import stl10_slic
import stanfordcars_slic
import geo_ds_slic
 
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from sklearn.model_selection import StratifiedKFold

import argparse


def load_dataset(n_splits, n_segments, compactness, features, graph_type, slic_method, dataset, pre_select_features):
    """
    loads dataset with specified parameters and returns:

    + dataset: torch's ConcatDataset with 1 or 2 datasets
    + splits: as returned by sklearn's StratifiedKFold split
    + labels: int list with dataset's labels 
    """
    ds = None
    if dataset == 'mnist':
        test_ds  = mnist_slic.SuperPixelGraphMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=False,
                                                   pre_select_features=pre_select_features)
        train_ds = mnist_slic.SuperPixelGraphMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=True,
                                                   pre_select_features=pre_select_features)
    elif dataset == 'fashion_mnist':
        test_ds  = fashion_mnist_slic.SuperPixelGraphFashionMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=False,
                                                   pre_select_features=pre_select_features)
        train_ds = fashion_mnist_slic.SuperPixelGraphFashionMNIST(root=None, 
                                                   n_segments=n_segments,
                                                   compactness=compactness,
                                                   features=features,
                                                   graph_type=graph_type,
                                                   slic_method=slic_method,
                                                   train=True,
                                                   pre_select_features=pre_select_features)
    elif dataset == 'cifar10':
        test_ds  = cifar10_slic.SuperPixelGraphCIFAR10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = cifar10_slic.SuperPixelGraphCIFAR10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'cifar100':
        test_ds  = cifar100_slic.SuperPixelGraphCIFAR100(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = cifar100_slic.SuperPixelGraphCIFAR100(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'stl10':
        test_ds  = stl10_slic.SuperPixelGraphSTL10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = stl10_slic.SuperPixelGraphSTL10(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'stanfordcars':
        test_ds  = stanfordcars_slic.SuperPixelGraphStanfordCars(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=False,
                                                       pre_select_features=pre_select_features)
        train_ds = stanfordcars_slic.SuperPixelGraphStanfordCars(root=None, 
                                                       n_segments=n_segments,
                                                       compactness=compactness,
                                                       features=features,
                                                       graph_type=graph_type,
                                                       slic_method=slic_method,
                                                       train=True,
                                                       pre_select_features=pre_select_features)
    elif dataset == 'geo_ds':
        ds = geo_ds_slic.SuperPixelGraphGeo('/home/julia/Documents/ds',
                                            root=None,
                                            n_segments=n_segments,
                                            compactness=compactness,
                                            features=features,
                                            graph_type=graph_type,
                                            slic_method=slic_method,
                                            pre_select_features=pre_select_features)
    else:
        print('No dataset called: \"' + dataset + '\" available.')
        return None
    if ds is None: 
        labels = train_ds.get_labels()
        targets = torch.cat([train_ds.get_targets(), test_ds.get_targets()])
        ds = ConcatDataset([train_ds, test_ds])
    else:
        labels = ds.get_labels()
        targets = ds.get_targets()
        ds = ConcatDataset([ds])
    splits = StratifiedKFold(n_splits=n_splits).split(np.zeros(len(targets)), targets)
    return ds, splits, labels

def set_dataset_arguments(parser):
    parser.add_argument("--n_splits", type=int, default=5,
                        help="number of splits in StratifiedKFold cross validation")
    parser.add_argument("--n_segments", type=int, default=75,
                        help="aproximate number of graph nodes. (default: 75)")
    parser.add_argument("--compactness", type=float, default=0.1,
                        help="compactness for SLIC algorithm. (default: 0.1)")
    parser.add_argument("--graph_type", type=str, default='RAG',
                        help="RAG, (1 | 2 | 4 | 8 | 16)NNSpatial or (1 | 2 | 4 | 8 | 16)NNFeatures")
    parser.add_argument("--slic_method", type=str, default='SLIC',
                        help="SLIC0, SLIC")
    parser.add_argument("--features", type=str, default=None,
                        help="space separated list of features. options are: avg_color, std_deviation_color, centroid, std_deviation_centroid, num_pixels. (default: avg_color centroid)")
    parser.add_argument("--dataset", default='mnist',
                        help="dataset to train against")
    parser.add_argument("--pre_select_features", action='store_true',
                        help="only save selected features when loading dataset")

    return parser

def get_dataset_params(args):
    ds_params = {}
    ds_params['n_splits'] = args.n_splits
    ds_params['n_segments'] = args.n_segments
    ds_params['compactness'] = args.compactness
    ds_params['graph_type'] = args.graph_type
    ds_params['slic_method'] = args.slic_method
    ds_params['dataset'] = args.dataset
    ds_params['pre_select_features'] = args.pre_select_features
    if args.features is not None:
        args.features = args.features.split()
    ds_params['features'] = args.features
    return ds_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = set_dataset_arguments(parser)
    args = parser.parse_args()
    ds_params = get_dataset_params(args)


    ds, splits, labels = load_dataset(ds_params['n_splits'],
                                      ds_params['n_segments'],
                                      ds_params['compactness'],
                                      ds_params['features'],
                                      ds_params['graph_type'], 
                                      ds_params['slic_method'],
                                      ds_params['dataset'],
                                      ds_params['pre_select_features'])
    
    print('Done.')
