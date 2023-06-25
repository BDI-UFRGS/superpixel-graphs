from color_slic import ColorSLIC

import torchvision.datasets as datasets 
import torchvision.transforms as T

class SuperPixelGraphStanfordCars(ColorSLIC):
    ds_name = 'StanfordCars'
    def get_ds_name(self):
        return  './stanfordcars/{}-n{}-{}-{}'.format('train' if self.train else 'test', 
                                                   self.n_segments, 
                                                   self.graph_type,
                                                   self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + str(self.compactness))
    def get_ds_name_with_features(self):
        self.features.sort()
        return  './stanfordcars/{}-n{}-{}-{}-{}'.format('train' if self.train else 'test', 
                                                        self.n_segments, 
                                                        self.graph_type,
                                                        self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + str(self.compactness),
                                                        '-'.join(self.features))
    def get_labels(self):
        return list(range(196))
    def load_data(self):
        split = 'train' if self.train else 'test'
        stanfordcars_root = f'./stanfordcars/{split}'
        return datasets.StanfordCars(stanfordcars_root, split=split, download=True, transform=T.ToTensor())