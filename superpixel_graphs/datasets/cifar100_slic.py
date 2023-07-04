from color_slic import ColorSLIC

import torchvision.datasets as datasets 
import torchvision.transforms as T

class SuperPixelGraphCIFAR100(ColorSLIC):
    ds_name = 'CIFAR100'
    def get_ds_name(self):
        self.features.sort()
        return  './cifar100/{}-n{}-{}-{}'.format('train' if self.train else 'test', 
                                                 self.n_segments, 
                                                 self.graph_type,
                                                 self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + str(self.compactness))
    def get_ds_name_with_features(self):
        self.features.sort()
        return  './cifar100/{}-n{}-{}-{}-{}'.format('train' if self.train else 'test', 
                                                    self.n_segments, 
                                                    self.graph_type,
                                                    self.slic_method if self.slic_method == 'SLIC0' else self.slic_method + 'c' + str(self.compactness),
                                                    '-'.join(self.features))
    def get_labels(self):
        return list(range(100))
    def load_data(self):
        cifar100_root = './cifar100/{}'.format('train' if self.train else 'test')
        return datasets.CIFAR100(cifar100_root, train=self.train, download=True, transform=T.ToTensor())