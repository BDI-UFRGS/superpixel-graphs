# Graph Neural Networks for image classification: comparing approaches for building graphs

In this folder are the experiments performed in order to evaluate how the graph-building process affects the performances of simple GCN and GAT models.

This work was developed in partial fulfillment of the requirements for the degree of Bachelor in Computer Science by Júlia Pelayo Rodrigues in UFRGS.

## Training a model 

Models are trained by running 

```
python train_model.py [args]
```

Help for defining the training parameters can be found by running 

```
python train_model.py --help
```

And for the graph building parameters for each available dataset run 
```
python dataset_loader.py --help
```

Datasets can also be computed individually by running 
```
python dataset_loader.py [args]
```

## Scripted experiments 

All experiments performed are listed in the [scripts folder](./scripts). 

## Visualizing results 

The [result_viz](./result_viz.ipynb) contains all the visualizations for the experiments. 

Size information for the models can be computed by running [get_model_size_info.py](./get-model-size-info.py).

Further information about datasets can be computed by running [get_dataset_info](./get_dataset_info.py).
