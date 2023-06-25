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
from torch.nn import Conv2d
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

import dataset_loader

class CNN(torch.nn.Module):
    def __init__(self, num_channels, img_size, num_classes):
        super(CNN, self).__init__()
        hidden_channel_size = 64 
        self.initial_conv = Conv2d(num_channels, hidden_channel_size, 3)
        self.conv1 = Conv2d(hidden_channel_size, hidden_channel_size, 3)
        self.conv2 = Conv2d(hidden_channel_size, hidden_channel_size, 3)
        self.out = nn.Linear(int(hidden_channel_size*2*((img_size-(2*3))/2)**2), num_classes)

    def forward(self, x):
        hidden = self.initial_conv(x)
        hidden = F.relu(hidden)
        hidden = self.conv1(hidden)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden)
        hidden = F.relu(hidden)
        hidden = torch.cat([F.max_pool2d(hidden, 2, 2),
                            F.avg_pool2d(hidden, 2, 2)], dim=1)
        hidden = torch.flatten(hidden, 1)
        out = self.out(hidden)
        return out 

def train(dataloader, model, loss_fn, optimizer, device):
    for _, b in enumerate(dataloader):
        x, y = b
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn, device, labels):
    num_batches = len(dataloader)
    test_loss = 0
    Y, Y_pred = torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for d in dataloader:
            x, y = d
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
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
    parser.add_argument("--dataset", type=str, 
                        help="dataset, choose from: mnist, fashion_mnist, cifar10, cifar100 or stl10")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    field_names = ["Epoch", 
                   "Accuracy", 
                   "Precision (micro)", "Precision (macro)", "Precision (weighted)", 
                   "Recall (micro)", "Recall (macro)", "Recall (weighted)", 
                   "F-measure (micro)", "F-measure (macro)", "F-measure (weighted)", 
                   "Avg loss"]
    meta_field_names = ['accuracy', 
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
    
    if args.dataset == 'mnist':
        train_ds = datasets.MNIST('./mnist/train', train=True, download=True, transform=T.ToTensor())
        test_ds = datasets.MNIST('./mnist/test', train=False, download=True, transform=T.ToTensor())
        num_classes = 10
        num_channels = 1
        img_size = 28
        targets = torch.cat((train_ds.targets, test_ds.targets))
        ds = ConcatDataset([train_ds, test_ds])
    elif args.dataset == 'fashion_mnist':
        train_ds = datasets.FashionMNIST('./fashion_mnist/train', train=True, download=True, transform=T.ToTensor())
        test_ds  = datasets.FashionMNIST('./fashion_mnist/test', train=False, download=True, transform=T.ToTensor())
        num_classes = 10
        num_channels = 1
        img_size = 28
        targets = torch.cat((train_ds.targets, test_ds.targets))
        ds = ConcatDataset([train_ds, test_ds])
    elif args.dataset == 'cifar10':
        train_ds = datasets.CIFAR10('./cifar10/train', train=True, download=True, transform=T.ToTensor())
        test_ds  = datasets.CIFAR10('./cifar10/test', train=False, download=True, transform=T.ToTensor())
        num_classes = 10
        num_channels = 3
        img_size = 32
        targets = torch.cat((torch.tensor(train_ds.targets), torch.tensor(test_ds.targets)))
        ds = ConcatDataset([train_ds, test_ds])
    elif args.dataset == 'cifar100':
        train_ds = datasets.CIFAR100('./cifar100/train', train=True, download=True, transform=T.ToTensor())
        test_ds  = datasets.CIFAR100('./cifar100/test', train=False, download=True, transform=T.ToTensor())
        num_classes = 100
        num_channels = 3
        img_size = 32
        targets = torch.cat((torch.tensor(train_ds.targets), torch.tensor(test_ds.targets)))
        ds = ConcatDataset([train_ds, test_ds])
    elif args.dataset == 'stl10':
        train_ds = datasets.STL10('./stl10/train', split='train', download=True, transform=T.ToTensor())
        test_ds  = datasets.STL10('./stl10/test', split='test', download=True, transform=T.ToTensor())
        num_classes = 10
        num_channels = 3
        img_size = 96
        targets = torch.cat((torch.from_numpy(train_ds.labels), torch.from_numpy(test_ds.labels)))
        ds = ConcatDataset([train_ds, test_ds])
    else:
        ds = None
    splits = StratifiedKFold(n_splits=5).split(np.zeros(len(targets)), targets)

    out = './{}/cnn.csv'.format(args.dataset)
    meta_out = './{}/cnn_training_info.csv'.format(args.dataset)

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

        model = CNN(num_channels, img_size, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        fold_hist = []
        print('------------------------')
        print(f'FOLD {len(history) + 1}/{5}')
        t0 = time.time()
        for t in range(epochs):
            train(train_loader, model, loss_fn, optimizer, device)
            res = test(test_loader, model, loss_fn, device, targets)
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

    meta_info = {}
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
