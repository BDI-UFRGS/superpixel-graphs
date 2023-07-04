import torch
from torch.utils.data import ConcatDataset, SubsetRandomSampler
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as T
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

import dataset_loader

class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        # using architecture inspired by MNISTSuperpixels example 
        # (https://medium.com/@rtsrumi07/understanding-graph-neural-network-with-hands-on-example-part-2-139a691ebeac)
        hidden_channel_size = 64 
        self.initial_conv = GCNConv(data.num_features, hidden_channel_size)
        self.conv1 = GCNConv(hidden_channel_size, hidden_channel_size)
        self.conv2 = GCNConv(hidden_channel_size, hidden_channel_size)
        self.out = nn.Linear(hidden_channel_size*2, data.num_classes)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = torch.cat([global_mean_pool(hidden, batch_index),
                            global_max_pool(hidden, batch_index)], dim=1)
        out = self.out(hidden)
        return out 

def train(dataloader, model, loss_fn, optimizer, device):
    for _, b in enumerate(dataloader):
        if type(b.y) != torch.Tensor:
            b.y = torch.tensor([b.y])
        b.y = b.y.type(torch.LongTensor)
        b.to(device)
        pred = model(b.x, b.edge_index, b.batch)
        loss = loss_fn(pred, b.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn, device, labels):
    num_batches = len(dataloader)
    test_loss = 0
    Y, Y_pred = torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for d in dataloader:
            if type(d.y) != torch.Tensor:
                d.y = torch.tensor([d.y])
            d.y = d.y.type(torch.LongTensor)
            d.to(device)
            pred = model(d.x, d.edge_index, d.batch)
            test_loss += loss_fn(pred, d.y).item()
            y = d.y
            if type(y) != torch.Tensor:
                y = torch.tensor([y])
            y = y.type(torch.LongTensor)
            Y = torch.cat([Y, y.to('cpu')])
            Y_pred = torch.cat([Y_pred, pred.to('cpu')])
    test_loss /= num_batches
    Y_pred = torch.argmax(Y_pred, dim=1)
    accuracy = accuracy_score(Y, Y_pred)
    precision_micro = precision_score(Y, Y_pred, average='micro', labels=labels, zero_division=0)
    precision_macro = precision_score(Y, Y_pred, average='macro', labels=labels, zero_division=0)
    precision_weighted = precision_score(Y, Y_pred, average='weighted', labels=labels, zero_division=0)
    recall_micro = recall_score(Y, Y_pred, average='micro', labels=labels, zero_division=0)
    recall_macro = recall_score(Y, Y_pred, average='macro', labels=labels, zero_division=0)
    recall_weighted = recall_score(Y, Y_pred, average='weighted', labels=labels, zero_division=0)
    f1_micro = f1_score(Y, Y_pred, average='micro', labels=labels)
    f1_macro = f1_score(Y, Y_pred, average='macro', labels=labels)
    f1_weighted = f1_score(Y, Y_pred, average='weighted', labels=labels)
    return {"Accuracy": accuracy, 
            "Precision (micro)": precision_micro, "Precision (macro)": precision_macro, "Precision (weighted)": precision_weighted,
            "Recall (micro)": recall_micro, "Recall (macro)": recall_macro, "Recall (weighted)": recall_weighted,
            "F-measure (micro)": f1_micro, "F-measure (macro)": f1_macro, "F-measure (weighted)": f1_weighted,
            "Avg loss": test_loss}

if __name__ == '__main__':
    import argparse
    import csv
    import time
    import os.path as osp

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="model's learning rate")
    parser.add_argument("--quiet", action="store_true")
    parser = dataset_loader.set_dataset_arguments(parser)
    args = parser.parse_args()

    field_names = ["Epoch", 
                   "Accuracy", 
                   "Precision (micro)", "Precision (macro)", "Precision (weighted)", 
                   "Recall (micro)", "Recall (macro)", "Recall (weighted)", 
                   "F-measure (micro)", "F-measure (macro)", "F-measure (weighted)", 
                   "Avg loss"]
    meta_field_names = ['n_segments', 
                        'compactness', 
                        'graph type', 
                        'slic method',
                        'features', 
                        'avg. num. of nodes', 
                        'std. dev. of num. of nodes', 
                        'avg. num. of edges', 
                        'std. dev. of num. of edges', 
                        'accuracy', 
                        'precision micro',
                        'precision macro',
                        'precision weighted',
                        'recall micro',
                        'recall macro',
                        'recall weighted',
                        'micro', 
                        'macro',
                        'weighted', 
                        'avg. loss', 
                        'training time',
                        'loading time']
    ds_params = dataset_loader.get_dataset_params(args)
    ds, splits, labels = dataset_loader.load_dataset(ds_params['n_splits'],
                                                     ds_params['n_segments'],
                                                     ds_params['compactness'],
                                                     ds_params['features'],
                                                     ds_params['graph_type'], 
                                                     ds_params['slic_method'],
                                                     ds_params['dataset'],
                                                     ds_params['pre_select_features'])
    meta_info = {}
    info_ds = ds.datasets[0]
    meta_info['loading time'] = info_ds.loading_time
    meta_info['avg. num. of nodes'] = info_ds.avg_num_nodes
    meta_info['std. dev. of num. of nodes'] = info_ds.std_deviation_num_nodes
    meta_info['avg. num. of edges'] = info_ds.avg_num_edges
    meta_info['std. dev. of num. of edges'] = info_ds.std_deviation_num_edges
    meta_info['n_segments']  = info_ds.n_segments
    meta_info['compactness'] = info_ds.compactness
    meta_info['graph type'] =  info_ds.graph_type
    meta_info['slic method'] = info_ds.slic_method
    meta_info['features'] = ' '.join(info_ds.features)
    
    out = './{}/n{}-{}-{}-{}.csv'.format(ds_params['dataset'],
                                         info_ds.n_segments,
                                         info_ds.graph_type,
                                         info_ds.slic_method if info_ds.slic_method == 'SLIC0' else info_ds.slic_method + 'c' + str(info_ds.compactness),
                                         '-'.join(info_ds.features))

    meta_out = './{}/training_info.csv'.format(ds_params['dataset'])
    with open(out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.manual_seed(42)

    epochs = args.epochs
    quiet = args.quiet

    history = []
    training_time = []
    for train_index, test_index in splits:
        train_loader = DataLoader(ds, batch_size=64, sampler=SubsetRandomSampler(train_index))
        test_loader  = DataLoader(ds, batch_size=64, sampler=SubsetRandomSampler(test_index))

        model = GCN(info_ds).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        fold_hist = []
        print('------------------------')
        print(f'FOLD {len(history) + 1}/{5}')
        t0 = time.time()
        for t in range(epochs):
            train(train_loader, model, loss_fn, optimizer, device)
            res = test(test_loader, model, loss_fn, device, labels)
            res["Epoch"] = t
            if not quiet:
                print(f'Epoch: {res["Epoch"]}, accuracy: {res["Accuracy"]}, loss: {res["Avg loss"]}')
            fold_hist.append(res)
        tf = time.time()
        print(f"Done in {tf - t0}s. Accuracy {fold_hist[-1]['Accuracy']}")
        training_time.append(tf - t0)
        history.append(fold_hist)

    avg_res = {}
    with open(out, 'a', newline='') as csvfile:
        history = np.array(history)
        for e in range(epochs):
            for field in field_names:
                avg_res[field] = np.average([f[field] for f in history[:,e]])
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(avg_res)

    meta_info['training time'] = np.average(training_time)
    meta_info['accuracy'] = avg_res['Accuracy']
    meta_info['precision micro'] = avg_res['Precision (micro)']
    meta_info['precision macro'] = avg_res['Precision (macro)']
    meta_info['precision weighted'] = avg_res['Precision (weighted)']
    meta_info['recall micro'] = avg_res['Recall (micro)']
    meta_info['recall macro'] = avg_res['Recall (macro)']
    meta_info['recall weighted'] = avg_res['Recall (weighted)']
    meta_info['micro'] = avg_res['F-measure (micro)']
    meta_info['macro'] = avg_res['F-measure (macro)']
    meta_info['weighted'] = avg_res['F-measure (weighted)']
    meta_info['avg. loss'] = avg_res['Avg loss']
    
    with open(meta_out, 'a', newline='') as infofile:
        writer = csv.DictWriter(infofile, fieldnames=meta_field_names)
        writer.writerow(meta_info)
