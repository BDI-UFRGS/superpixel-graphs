import dataset_loader as dsl

if __name__ == '__main__':
    import argparse
    import csv 

    parser = argparse.ArgumentParser()
    parser = dsl.set_dataset_arguments(parser)
    args = parser.parse_args()
    ds_params = dsl.get_dataset_params(args)

    ds, _, _ = dsl.load_dataset(ds_params['n_splits'],
                                ds_params['n_segments'],
                                ds_params['compactness'],
                                ds_params['features'],
                                ds_params['graph_type'], 
                                ds_params['slic_method'],
                                ds_params['dataset'],
                                ds_params['pre_select_features'])

    fields = ['n_segments', 
              'compactness',
              'graph type',
              'slic method',
              'features',
              'avg. num. of nodes',
              'std. dev. of num. of nodes', 
              'avg. num. of edges', 
              'std. dev. of num. of edges']

    out = './{}/dataset_info.csv'.format(ds_params['dataset'])
    
    info_ds = ds.datasets[0]
    info = {}
    info['n_segments']  = info_ds.n_segments
    info['compactness'] = info_ds.compactness
    info['graph type'] =  info_ds.graph_type
    info['slic method'] = info_ds.slic_method
    info['features'] = ' '.join(info_ds.features)
    info['avg. num. of nodes'] = info_ds.avg_num_nodes
    info['std. dev. of num. of nodes'] = info_ds.std_deviation_num_nodes
    info['avg. num. of edges'] = info_ds.avg_num_edges
    info['std. dev. of num. of edges'] = info_ds.std_deviation_num_edges
    
    with open(out, 'a', newline='') as infofile:
        writer = csv.DictWriter(infofile, fieldnames=fields)
        writer.writerow(info)
