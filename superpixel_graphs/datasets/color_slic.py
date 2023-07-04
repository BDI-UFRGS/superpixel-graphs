import torch
import torchvision.datasets as datasets 
import torchvision.transforms as T
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import rgb2hsv
import skimage as ski
import time

try:
    from compute_features import color_features
except ImportError:
    extension_availabe = False
else:
    extension_availabe = True


class ColorSLIC(InMemoryDataset):
    std_features = ['avg_color',
                    'std_deviation_color',
                    'centroid',
                    'std_deviation_centroid']
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

    ds_name = 'ColorSLIC'
    def __init__(self, 
                 root=None, 
                 n_segments= 75,  
                 compactness = 0.1, 
                 features = None, 
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
        self.select_features()
        transform = None if self.pre_select_features else self.filter_features
        super().__init__(root=self.root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
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
    
    def load_data(self):
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
        self.add_feature('avg_color_hsv')
        self.add_feature('std_deviation_color_hsv')
        self.print_features()
    
    def add_feature(self, feature):
        f = feature in self.features
        if 'color' in feature:
            self.features_mask.append(f)
            self.features_mask.append(f)
            self.features_mask.append(f)
        elif 'centroid' in feature:
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
            if self.pre_select_features:
                data_list = [self.filter_features(self.create_data_obj_ext(d)) for d in data]
            else:
                data_list = [self.create_data_obj_ext(d) for d in data]
        else:
            if self.pre_select_features:
                data_list = [self.filter_features(self.create_data_obj(d)) for d in data]
            else:
                data_list = [self.create_data_obj(d) for d in data]
        t = time.time() - t
        self.loading_time = t
        print(f'Done in {t}s')
        self.save_stats(data_list)
        return self.collate(data_list)

    def create_data_obj_ext(self, d):
            img, y = d
            img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
            features, edge_index, _ = color_features(img_np,
                                                     self.n_segments, 
                                                     self.graph_types_dict[self.graph_type], 
                                                     self.slic_methods_dict[self.slic_method], 
                                                     self.compactness)
            pos = features[:, 6:8]
            return Data(x=torch.from_numpy(features).to(torch.float), edge_index=torch.from_numpy(edge_index).to(torch.long), pos=torch.from_numpy(pos).to(torch.float), y=y)

    def create_data_obj(self, d):
            img, y = d
            _, dim0, dim1 = img.shape
            img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
            img_hsv = rgb2hsv(img_np)
            s = slic(img_np, self.n_segments, self.compactness, start_label=0)
            # rag_mean_color() fails when image is segmented into 1 superpixel 
            if np.any(s):
                g = ski.future.graph.rag_mean_color(img_np, s)
                n = g.number_of_nodes()
                edge_index = torch.from_numpy(np.array(g.edges).T).to(torch.long)
            else: 
                n = 1
                edge_index = torch.tensor([]).to(torch.long)
            s1 = np.zeros([n, 3])  # for mean color and std deviation
            s2 = np.zeros([n, 3])  # for std deviation
            s1_hsv = np.zeros([n,3])
            s2_hsv = np.zeros([n,3])
            pos1 = np.zeros([n, 2]) # for centroid
            pos2 = np.zeros([n, 2]) # for centroid std deviation
            num_pixels = np.zeros([n, 1])
            for idx in range(dim0 * dim1):
                    idx_i, idx_j = idx % dim0, int(idx / dim0)
                    node = s[idx_i][idx_j] - 1
                    s1[node][0]  += img_np[idx_i][idx_j][0]
                    s2[node][0]  += pow(img_np[idx_i][idx_j][0], 2)
                    s1[node][1]  += img_np[idx_i][idx_j][1]
                    s2[node][1]  += pow(img_np[idx_i][idx_j][1], 2)
                    s1[node][2]  += img_np[idx_i][idx_j][2]
                    s2[node][2]  += pow(img_np[idx_i][idx_j][2], 2)
                    s1_hsv[node][0]  += img_hsv[idx_i][idx_j][0]
                    s2_hsv[node][0]  += pow(img_hsv[idx_i][idx_j][0], 2)
                    s1_hsv[node][1]  += img_hsv[idx_i][idx_j][1]
                    s2_hsv[node][1]  += pow(img_hsv[idx_i][idx_j][1], 2)
                    s1_hsv[node][2]  += img_hsv[idx_i][idx_j][2]
                    s2_hsv[node][2]  += pow(img_hsv[idx_i][idx_j][2], 2)
                    pos1[node][0] += idx_i
                    pos1[node][1] += idx_j
                    pos2[node][0] += pow(idx_i, 2)
                    pos2[node][1] += pow(idx_j, 2)
                    num_pixels[node][0] += 1
            x = []
            s1 = s1/num_pixels
            avg_color = torch.from_numpy(s1).to(torch.float)
            x.append(avg_color[:,0])
            x.append(avg_color[:,1])
            x.append(avg_color[:,2])
            s2 = s2/num_pixels
            std_dev = torch.from_numpy(np.sqrt(np.abs((s2 - s1*s1)))).to(torch.float)
            x.append(std_dev[:,0])
            x.append(std_dev[:,1])
            x.append(std_dev[:,2])
            s1_hsv = s1_hsv/num_pixels
            avg_color_hsv = torch.from_numpy(s1_hsv).to(torch.float)
            x.append(avg_color_hsv[:,0])
            x.append(avg_color_hsv[:,1])
            x.append(avg_color_hsv[:,2])
            s2_hsv = s2_hsv/num_pixels
            std_dev_hsv = torch.from_numpy(np.sqrt(np.abs((s2_hsv - s1_hsv*s1_hsv)))).to(torch.float)
            x.append(std_dev_hsv[:,0])
            x.append(std_dev_hsv[:,1])
            x.append(std_dev_hsv[:,2])
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
    
    def get_og_img(self, idx):
        data = self.load_data()
        img, _ = data[idx]
        img_np = torch.stack([img[0], img[1], img[2]], dim=2).numpy()
        return img_np

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data, slices = self.load()
        torch.save((data, slices), self.processed_paths[0])