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
    from compute_features import grayscale_features
except ImportError:
    extension_availabe = False
else:
    extension_availabe = True

class GrayscaleSLIC(InMemoryDataset):
    # base class for grayscale datasets 
    # children must implement 
    # get_ds_name: returns a string with the path to the dataset
    # load_data: ...
    # optionally ds_name for pretty printing 

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

    std_features = ['avg_color',
                    'std_deviation_color',
                    'centroid',
                    'std_deviation_centroid']
    
    ds_name = 'GrayscaleSLIC'
    
    def __init__(self, 
                 root=None, 
                 n_segments= 75,  
                 compactness = 0.1, 
                 features = None, # possible features are avg_color, centroid, std_deviation_color 
                 graph_type = 'RAG',
                 slic_method = 'SLIC0',
                 train = True,
                 pre_select_features = False):
        self.train = train
        self.n_segments = n_segments
        self.compactness = compactness
        self.features = self.std_features if features is None else features
        self.graph_type = graph_type
        self.slic_method = slic_method
        self.pre_select_features = pre_select_features

        if root is None:
            self.root = self.get_ds_name_with_features() if self.pre_select_features else self.get_ds_name()
        else:
            self.root = root

        self.is_pre_loaded = True
        super().__init__(root=self.root, transform=self.filter_features)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        self.select_features()

        self.get_stats()
        print(self.ds_name + " Loaded.")
        print(f"Average number of nodes: {self.avg_num_nodes} with standard deviation {self.std_deviation_num_nodes}")
        print(f"Average number of edges: {self.avg_num_edges} with standard deviation {self.std_deviation_num_edges}")

    def get_ds_name(self):
        raise NotImplementedError
    
    def get_ds_name_with_features(self):
        raise NotImplementedError
    
    def get_labels(self):
        raise NotImplementedError
    
    def get_targets(self):
        return torch.cat([d.y for d in self])
    
    def select_features(self):
        self.features_mask = []
        self.features_dict = {}
        self.add_feature('avg_color')
        self.add_feature('std_deviation_color')
        self.add_feature('centroid')
        self.add_feature('std_deviation_centroid')
        self.add_feature('num_pixels')
        self.print_features()

    def add_feature(self, feature):
        f = feature in self.features
        if 'centroid' in feature:
            self.features_mask.append(f)
            self.features_mask.append(f)
        else:
            self.features_mask.append(f)
        self.features_dict[feature] = f

    def print_features(self):
        print('Selected features for ' + self.graph_type + ' graph:')
        for feature in self.features_dict:
            if self.features_dict[feature]:
                print('\t+ ' + feature)
    
    def filter_features(self, data):
        x_trans = data.x.numpy()
        x_trans = x_trans[:, self.features_mask]
        data.x = torch.from_numpy(x_trans).to(torch.float)
        return data

    def load(self):
        self.is_pre_loaded = False
        data = self.load_data()
        img_total = len(data)
        print(f'Loading {img_total} images with n_segments = {self.n_segments} ...')
        print(f'Computing features: ')

        t = time.time()
        if extension_availabe:
            data_list = [self.create_data_obj_ext(d) for d in data]
        else:
            data_list = [self.create_data_obj(d) for d in data]
        t = time.time() - t
        self.loading_time = t
        print(f'Done in {t}s')
        self.save_stats(data_list)
        return self.collate(data_list)

    def load_data(self):
        raise NotImplementedError

    def create_data_obj_ext(self, d):
            img, y = d
            _, dim0, dim1 = img.shape
            img_np = img.view(dim0, dim1).numpy()
            features, edge_index, _ = grayscale_features(img_np, 
                                                         self.n_segments, 
                                                         self.graph_types_dict[self.graph_type], 
                                                         self.slic_methods_dict[self.slic_method], 
                                                         self.compactness)
            pos = features[:, 2:4]
            return Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), y=y)

    def create_data_obj(self, d):
            img, y = d
            _, dim0, dim1 = img.shape
            img_np = img.view(dim0, dim1).numpy()
            s = slic(img_np, self.n_segments, self.compactness, start_label=0)
            if np.any(s):
                g = ski.future.graph.rag_mean_color(img_np, s)
                n = g.number_of_nodes()
                edge_index = torch.from_numpy(np.array(g.edges).T).to(torch.long)
            else:
                n = 1
                edge_index = torch.tensor([]).to(torch.long)
            s1 = np.zeros([n, 1])  # for mean color and std deviation
            s2 = np.zeros([n, 1])  # for std deviation
            pos1 = np.zeros([n, 2]) # for centroid
            pos2 = np.zeros([n, 2]) # for centroid std deviation
            num_pixels = np.zeros([n, 1])
            for idx in range(dim0 * dim1):
                    idx_i, idx_j = idx % dim0, int(idx / dim0)
                    node = s[idx_i][idx_j] - 1
                    s1[node][0]  += img_np[idx_i][idx_j]
                    s2[node][0]  += pow(img_np[idx_i][idx_j], 2)
                    pos1[node][0] += idx_i
                    pos1[node][1] += idx_j
                    pos2[node][0] += pow(idx_i, 2)
                    pos2[node][1] += pow(idx_j, 2)
                    num_pixels[node][0] += 1
            x = []
            s1 = s1/num_pixels
            x.append(torch.from_numpy(s1.flatten()).to(torch.float))
            s2 = s2/num_pixels
            std_dev = np.sqrt(np.abs((s2 - s1*s1)))
            x.append(torch.from_numpy(std_dev.flatten()).to(torch.float))
            pos1 = pos1/num_pixels
            pos = torch.from_numpy(pos1).to(torch.float)
            x.append(pos[:,0])
            x.append(pos[:,1])
            pos2 = pos2/num_pixels
            std_dev_centroid = torch.from_numpy(np.sqrt(np.abs(pos2 - pos1*pos1))).to(torch.float)
            x.append(std_dev_centroid[:,0])
            x.append(std_dev_centroid[:,1])
            x.append(torch.from_numpy(num_pixels.flatten()).to(torch.float))
            return Data(x=torch.stack(x, dim=1), edge_index=edge_index, pos=pos, y=y)

    def save_stats(self, data):
        nodes = [d.num_nodes for d in data]
        edges = [d.num_edges for d in data]
        self.avg_num_nodes = np.average(nodes)
        self.std_deviation_num_nodes = np.std(nodes)
        self.avg_num_edges = np.average(edges)
        self.std_deviation_num_edges = np.std(edges)
    
    def get_stats(self):
        if self.is_pre_loaded:
            data_list = [self[i] for i in range(len(self))]
            self.save_stats(data_list)
            self.loading_time = 0

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data, slices = self.load()
        torch.save((data, slices), self.processed_paths[0])
    