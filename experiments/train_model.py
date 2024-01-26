from models import GCN, GAT, CNN, train, eval
from torchvision.models.efficientnet import EfficientNet, efficientnet_b0
import dataset_loader as dsl

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, Subset
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

random_seed = 42

def checkpoint(model, directory, filename, fold):
    file = directory + filename + f'.fold{fold}' + '.pth'
    torch.save(model.state_dict(), file)

if __name__ == '__main__':
    import argparse
    import csv
    import time
    import os

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="GCN", 
                        help="the model to train: GCN, GAT, CNN, AlexNet, EfficientNet. default = GCN")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="max number of training epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--patience", type=int, default=-1,
                        help="allowed epochs without min. improvement berfore early stopping, negative number disables early stopping. default = -1, no early stopping.")
    parser.add_argument("--min_improvement", type=float, default=0.001,
                        help="min improvement from previous epoch. default = 0.001")
    parser.add_argument("--validation_size", type=float, default=0.1,
                        help="fraction of validation data in the train/validation split")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--n_heads", type=int, default=4, 
                        help='number of attention heads in GAT layer. default = 4')
    parser.add_argument('--n_layers', type=int, default=2, 
                        help='number of stacked conv. layers (GAT or GCN). default = 2')
    parser.add_argument('--info_filename', '-f', type=str, default='training_info', 
                        help='name of file where training information is stored')
    parser = dsl.set_dataset_arguments(parser)
    args = parser.parse_args()

    field_names = ["epoch", 
                   "accuracy", 
                   "precision (micro)", "precision (macro)", "precision (weighted)", 
                   "recall (micro)", "recall (macro)", "recall (weighted)", 
                   "f1-measure (micro)", "f1-measure (macro)", "f1-measure (weighted)", 
                   "validation f1-measure (micro)", "validation f1-measure (macro)", "validation f1-measure (weighted)", 
                   "train f1-measure (micro)", "train f1-measure (macro)", "train f1-measure (weighted)", 
                   "loss", "validation loss", "train loss"]
    meta_field_names = ['model',
                        'num. layers',
                        'num. heads',
                        'stopped-at',
                        'n_segments', 
                        'compactness', 
                        'graph type', 
                        'slic method',
                        'features', 
                        'avg. num. of nodes', 
                        'std. dev. of num. of nodes', 
                        'avg. num. of edges', 
                        'std. dev. of num. of edges', 
                        'best epochs',
                        'last epochs',
                        'accuracy', 'stdv accuracy', 
                        'precision micro', 'stdv precision micro',
                        'precision macro', 'stdv precision macro',
                        'precision weighted', 'stdv precision weighted',
                        'recall micro', 'stdv recall micro',
                        'recall macro', 'stdv recall macro',
                        'recall weighted', 'stdv recall weighted',
                        'f1 micro', 'stdv f1 micro', 
                        'f1 macro', 'stdv f1 macro',
                        'f1 weighted', 'stdv f1 weighted', 
                        'validation f1 micro', 'stdv validation f1 micro', 
                        'validation f1 macro', 'stdv validation f1 macro',
                        'validation f1 weighted', 'stdv validation f1 weighted', 
                        'train f1 micro', 'stdv train f1 micro', 
                        'train f1 macro', 'stdv train f1 macro',
                        'train f1 weighted', 'stdv train f1 weighted', 
                        'loss', 'stdv loss', 
                        'validation loss', 'stdv validation loss', 
                        'train loss', 'stdv train loss', 
                        'training time', 'stdv training time',
                        'loading time', 'stdv loading time']
    t0 = time.time()
    ds, splits, targets = dsl.load_dataset(args)
    loading_time = time.time() - t0
    ds_info = dsl.dataset_info(args)

    meta_info = {}
    info_ds = ds.datasets[0]
    meta_info['model'] = args.model
    meta_info['loading time'] = loading_time
    if args.model not in ['CNN', 'AlexNet', 'EfficientNet']:
        meta_info['num. layers'] = args.n_layers
        if args.model == 'GAT':
            meta_info['num. heads'] = args.n_heads
        else:
            meta_info['num. heads'] = '-' 
        meta_info['avg. num. of nodes'] = info_ds.avg_num_nodes
        meta_info['std. dev. of num. of nodes'] = info_ds.std_deviation_num_nodes
        meta_info['avg. num. of edges'] = info_ds.avg_num_edges
        meta_info['std. dev. of num. of edges'] = info_ds.std_deviation_num_edges
        meta_info['n_segments']  = info_ds.n_segments
        meta_info['graph type'] =  info_ds.graph_type
        meta_info['slic method'] = info_ds.slic_method
        meta_info['features'] = ' '.join(info_ds.features)
        if info_ds.slic_method == 'SLIC':
            meta_info['compactness'] = info_ds.compactness
        else:
            meta_info['compactness'] = '-'
    else:
        meta_info['num. layers'] = 3
        meta_info['num. heads'] = '-' 
        meta_info['avg. num. of nodes'] = '-'
        meta_info['std. dev. of num. of nodes'] = '-'
        meta_info['avg. num. of edges'] = '-'
        meta_info['std. dev. of num. of edges'] = '-'
        meta_info['n_segments']  = '-'
        meta_info['compactness'] = '-'
        meta_info['graph type'] = '-'
        meta_info['slic method'] = '-'
        meta_info['features'] = '-'
    
    out_dir = f'./{args.model}/{args.dataset}/'
    if args.model in ['CNN',  'AlexNet', 'EfficientNet']:
        out_file = '{}-lr{}'.format(args.model, args.learning_rate)
    elif args.model == 'GAT':
        out_file = 'l{}h{}n{}-{}-{}-{}'.format(args.n_layers, 
                                                   args.n_heads, 
                                                   info_ds.n_segments,
                                                   info_ds.graph_type,
                                                   info_ds.slic_method if info_ds.slic_method == 'SLIC0' else info_ds.slic_method + 'c' + str(info_ds.compactness),
                                                   '-'.join(info_ds.features))
    else:
        out_file = 'l{}n{}-{}-{}-{}'.format(args.n_layers, 
                                                info_ds.n_segments,
                                                info_ds.graph_type,
                                                info_ds.slic_method if info_ds.slic_method == 'SLIC0' else info_ds.slic_method + 'c' + str(info_ds.compactness),
                                                '-'.join(info_ds.features))


    meta_out = './{}.csv'.format(args.info_filename)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + out_file + '.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    epochs = args.epochs
    verbose_output = args.verbose
    patience = args.patience
    min_improvement = args.min_improvement 

    history = []
    training_time = []
    last_epochs = []
    best_epochs = []
    best_results = []

    # stratified k-fold cross validation 
    for train_validation_index, test_index in splits:
        # test data 
        test_loader  = DataLoader(ds, batch_size=64, sampler=SubsetRandomSampler(test_index))
        
        # train data divided into validation_size% validation and (1-validation_size)% train, maintaning class proportions 
        sss = StratifiedShuffleSplit(n_splits=1, 
                                     test_size=args.validation_size,
                                     random_state=random_seed)
        train_index, validation_index = sss.split(np.zeros(len(train_validation_index)),
                                                  Subset(targets, train_validation_index)).__next__()
        train_validation_ds = Subset(ds, train_validation_index)
        train_loader = DataLoader(train_validation_ds, batch_size=64, sampler=SubsetRandomSampler(train_index))
        validation_loader = DataLoader(train_validation_ds, batch_size=64, sampler=SubsetRandomSampler(validation_index))

        if args.model == 'GCN':
            model = GCN(info_ds.num_features, ds_info['classes'], args.n_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            loss_fn = torch.nn.CrossEntropyLoss()
        elif args.model == 'GAT':
            model = GAT(info_ds.num_features, ds_info['classes'], args.n_heads, args.n_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            loss_fn = torch.nn.CrossEntropyLoss()
        elif args.model == CNN:
            model = CNN(ds_info['channels'], ds_info['classes'], ds_info['img_size']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            loss_fn = torch.nn.CrossEntropyLoss()
        elif args.model == 'AlexNet':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=None).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            loss_fn = torch.nn.CrossEntropyLoss()
        elif args.model == 'EfficientNet':
            model = efficientnet_b0().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            print(f"No dataset named \"{args.dataset}\" available.")

        fold_hist = []
        epochs_without_improvement = 0
        best_validation_res = {}
        best_test_res = {}
        best_epoch = 0
        last_epoch = 0
        print('------------------------')
        print(f'FOLD {len(history) + 1}/{5}')
        t0 = time.time()
        for t in range(epochs):
            # 1. train model 
            train(train_loader, model, loss_fn, optimizer, device)
            train_res = eval(train_loader, model, loss_fn, device, targets)

            # 2. evaluate model with validation set, checking if model should stop 
            #    and keeping track of the best epoch so far 
            validation_res = eval(validation_loader, model, loss_fn, device, targets)
            if t > 0:
                if validation_res['loss'] - best_validation_res['loss'] > -min_improvement:
                    epochs_without_improvement += 1
                elif epochs_without_improvement > 0:
                    epochs_without_improvement = 0
                    
                if validation_res['loss'] < best_validation_res['loss']:
                    best_validation_res = validation_res
                    best_epoch = t
                    checkpoint(model, out_dir, out_file, len(history))
            else:
                best_validation_res = validation_res
                best_epoch = t
                checkpoint(model, out_dir, out_file, len(history))
            
            # 3. evaluate model with test set, reporting performance metrics 
            test_res = eval(test_loader, model, loss_fn, device, targets)
            test_res['epoch'] = t
            test_res['validation loss'] = validation_res['loss']
            test_res['validation f1-measure (micro)'] = validation_res['f1-measure (micro)']
            test_res['validation f1-measure (macro)'] = validation_res['f1-measure (macro)']
            test_res['validation f1-measure (weighted)'] = validation_res['f1-measure (weighted)']
            test_res['train loss'] = train_res['loss']
            test_res['train f1-measure (micro)'] = train_res['f1-measure (micro)']
            test_res['train f1-measure (macro)'] = train_res['f1-measure (macro)']
            test_res['train f1-measure (weighted)'] = train_res['f1-measure (weighted)']
            
            if best_epoch == t:
                best_test_res = test_res
            if verbose_output:
                print(f'Epoch: {t}, f1: {test_res["f1-measure (macro)"]}, loss: {test_res["loss"]}')
            fold_hist.append(test_res)

            # early stop
            if patience > 0 and epochs_without_improvement > patience:
                print(f'Stopped at epoch {t}')
                break

        tf = time.time()
        print(f"Done in {tf - t0}s. F-measure {fold_hist[-1]['f1-measure (macro)']}")
        training_time.append(tf - t0)
        history.append(fold_hist)
        last_epochs.append(str(fold_hist[-1]['epoch']))
        best_epochs.append(str(best_epoch))
        best_results.append(best_test_res)

    avg_result_epoch = {}
    for e in range(epochs):
        results = []
        for fold in range(len(history)):
            try:
                results.append(history[fold][e])
            except:
                continue
        for field in field_names:
            avg_result_epoch[field] = np.average([f[field] for f in results])
            if np.isnan(avg_result_epoch[field]):
                avg_result_epoch[field] = ''
        with open(out_dir + out_file + '.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(avg_result_epoch)

    final_result = {}
    for field in field_names:
        final_result[field] = [f[field] for f in best_results]
    meta_info['training time'] = np.average(training_time)
    meta_info['accuracy'] = np.average(final_result['accuracy'])
    meta_info['precision micro'] = np.average(final_result['precision (micro)'])
    meta_info['precision macro'] = np.average(final_result['precision (macro)'])
    meta_info['precision weighted'] = np.average(final_result['precision (weighted)'])
    meta_info['recall micro'] = np.average(final_result['recall (micro)'])
    meta_info['recall macro'] = np.average(final_result['recall (macro)'])
    meta_info['recall weighted'] = np.average(final_result['recall (weighted)'])
    meta_info['f1 micro']         = np.average(final_result['f1-measure (micro)'])
    meta_info['stdv f1 micro']    = np.std(final_result['f1-measure (micro)'])
    meta_info['f1 macro']         = np.average(final_result['f1-measure (macro)'])
    meta_info['stdv f1 macro']    = np.std(final_result['f1-measure (macro)'])
    meta_info['f1 weighted']      = np.average(final_result['f1-measure (weighted)'])
    meta_info['stdv f1 weighted'] = np.std(final_result['f1-measure (weighted)'])
    meta_info['validation f1 micro']         = np.average(final_result['validation f1-measure (micro)'])
    meta_info['stdv validation f1 micro']    = np.std(final_result['validation f1-measure (micro)'])
    meta_info['validation f1 macro']         = np.average(final_result['validation f1-measure (macro)'])
    meta_info['stdv validation f1 macro']    = np.std(final_result['validation f1-measure (macro)'])
    meta_info['validation f1 weighted']      = np.average(final_result['validation f1-measure (weighted)'])
    meta_info['stdv validation f1 weighted'] = np.std(final_result['validation f1-measure (weighted)'])
    meta_info['train f1 micro']         = np.average(final_result['train f1-measure (micro)'])
    meta_info['stdv train f1 micro']    = np.std(final_result['train f1-measure (micro)'])
    meta_info['train f1 macro']         = np.average(final_result['train f1-measure (macro)'])
    meta_info['stdv train f1 macro']    = np.std(final_result['train f1-measure (macro)'])
    meta_info['train f1 weighted']      = np.average(final_result['train f1-measure (weighted)'])
    meta_info['stdv train f1 weighted'] = np.std(final_result['train f1-measure (weighted)'])
    meta_info['loss']      = np.average(final_result['loss'])
    meta_info['stdv loss'] = np.std(final_result['loss'])
    meta_info['validation loss']      = np.average(final_result['validation loss'])
    meta_info['stdv validation loss'] = np.std(final_result['validation loss'])
    meta_info['train loss']      = np.average(final_result['train loss'])
    meta_info['stdv train loss'] = np.std(final_result['train loss'])
    meta_info['last epochs'] = ', '.join(last_epochs)
    meta_info['best epochs'] = ', '.join(best_epochs)


    if not os.path.isfile(meta_out):
        with open(meta_out, 'a', newline='') as infofile:
            writer = csv.DictWriter(infofile, fieldnames=meta_field_names)
            writer.writeheader()

    with open(meta_out, 'a', newline='') as infofile:
        writer = csv.DictWriter(infofile, fieldnames=meta_field_names)
        writer.writerow(meta_info)
