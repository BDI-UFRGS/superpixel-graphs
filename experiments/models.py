import torch
import torchvision 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torchvision.models.alexnet import AlexNet
from torchvision.models.efficientnet import EfficientNet, efficientnet_b0
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        # using architecture inspired by MNISTSuperpixels example 
        # (https://medium.com/@rtsrumi07/understanding-graph-neural-network-with-hands-on-example-part-2-139a691ebeac)
        hidden_channel_size = 64 
        self.initial_conv = GCNConv(in_channels, hidden_channel_size)
        self.hidden_layers = nn.ModuleList([])
        for _ in range(num_layers - 1):
            self.hidden_layers.append(GCNConv(hidden_channel_size, hidden_channel_size))
        self.out = nn.Linear(hidden_channel_size*2, out_channels)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        for hidden_layer in self.hidden_layers:
            hidden = hidden_layer(hidden, edge_index)
            hidden = F.relu(hidden)
        hidden = torch.cat([global_mean_pool(hidden, batch_index),
                            global_max_pool(hidden, batch_index)], dim=1)
        out = self.out(hidden)
        return out 

class GCNFeatures(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(GCNFeatures, self).__init__()
        # using architecture inspired by MNISTSuperpixels example 
        # (https://medium.com/@rtsrumi07/understanding-graph-neural-network-with-hands-on-example-part-2-139a691ebeac)
        hidden_channel_size = 64 
        self.initial_conv = GCNConv(in_channels, hidden_channel_size)
        self.hidden_layers = nn.ModuleList([])
        for _ in range(num_layers - 1):
            self.hidden_layers.append(GCNConv(hidden_channel_size, hidden_channel_size))

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        for hidden_layer in self.hidden_layers:
            hidden = hidden_layer(hidden, edge_index)
            hidden = F.relu(hidden)
        out = hidden
        return out 
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(GAT, self).__init__()
        in_out_size = 32
        out_size = 64
        self.initial_conv = GATConv(in_channels, in_out_size, heads=num_heads)
        in_size = in_out_size * num_heads
        self.hidden_layers = nn.ModuleList([]) 
        for _ in range(num_layers-1):
            self.hidden_layers.append(GATConv(in_size, out_size, heads=num_heads))
            in_size = out_size * num_heads
        self.out = nn.Linear(in_size * 2, out_channels)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        for gat in self.hidden_layers:
            hidden = gat(hidden, edge_index)
            hidden = F.relu(hidden)
        hidden = torch.cat([global_mean_pool(hidden, batch_index),
                            global_max_pool(hidden, batch_index)], dim=1)
        out = self.out(hidden)
        return out 

class GATFeatures(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(GATFeatures, self).__init__()
        in_out_size = 32
        out_size = 64
        self.initial_conv = GATConv(in_channels, in_out_size, heads=num_heads)
        in_size = in_out_size * num_heads
        self.hidden_layers = nn.ModuleList([]) 
        for _ in range(num_layers-1):
            self.hidden_layers.append(GATConv(in_size, out_size, heads=num_heads))
            in_size = out_size * num_heads

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        for gat in self.hidden_layers:
            hidden = gat(hidden, edge_index)
            hidden = F.relu(hidden)
        out = hidden
        return out 
    
class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, img_size):
        super(CNN, self).__init__()
        hidden_channel_size = 32 
        self.initial_conv = Conv2d(in_channels, hidden_channel_size, 3)
        self.conv1 = Conv2d(hidden_channel_size, hidden_channel_size, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(int(hidden_channel_size * ((img_size-4)/2)**2 ), 128)
        self.dropout2 = nn.Dropout1d(0.50)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        hidden = self.initial_conv(x)
        hidden = F.relu(hidden)
        hidden = self.conv1(hidden)
        hidden = F.relu(hidden)
        hidden = F.max_pool2d(hidden, 2, 2)
        hidden = self.dropout1(hidden)
        
        hidden = torch.flatten(hidden, 1)
        hidden = self.fc1(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.fc2(hidden)
        out = F.softmax(hidden, 1)

        return out 

def train(trainloader, model, loss_fn, optimizer, device):
    if type(model) in [CNN, AlexNet, EfficientNet]:
        train_images(model, trainloader, loss_fn, optimizer, device)
    else: # GCN, GAT
        train_graphs(model, trainloader, loss_fn, optimizer, device)

def train_graphs(model, trainloader, loss_fn, optimizer, device):
    for d in trainloader:
        if type(d.y) != torch.Tensor:
            d.y = torch.tensor([d.y])
        d.y = d.y.type(torch.LongTensor)
        d.to(device)
        pred = model(d.x, d.edge_index, d.batch)
        loss = loss_fn(pred, d.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_images(model, trainloader, loss_fn, optimizer, device):
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval(testloader, model, loss_fn, device, labels):
    if type(model) in [CNN, AlexNet, EfficientNet]:
        return eval_images(testloader, model, loss_fn, device, labels)
    else:
        return eval_graphs(testloader, model, loss_fn, device, labels)
    
def eval_graphs(testloader, model, loss_fn, device, labels):
    num_batches = len(testloader)
    test_loss = 0
    Y, Y_pred = torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for d in testloader:
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
    return {"accuracy": accuracy, 
            "precision (micro)": precision_micro, "precision (macro)": precision_macro, "precision (weighted)": precision_weighted,
            "recall (micro)": recall_micro, "recall (macro)": recall_macro, "recall (weighted)": recall_weighted,
            "f1-measure (micro)": f1_micro, "f1-measure (macro)": f1_macro, "f1-measure (weighted)": f1_weighted,
            "loss": test_loss}

def eval_images(testloader, model, loss_fn, device, labels):
    num_batches = len(testloader)
    test_loss = 0
    Y, Y_pred = torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for d in testloader:
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
    return {"accuracy": accuracy, 
            "precision (micro)": precision_micro, "precision (macro)": precision_macro, "precision (weighted)": precision_weighted,
            "recall (micro)": recall_micro, "recall (macro)": recall_macro, "recall (weighted)": recall_weighted,
            "f1-measure (micro)": f1_micro, "f1-measure (macro)": f1_macro, "f1-measure (weighted)": f1_weighted,
            "loss": test_loss}