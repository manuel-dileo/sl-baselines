# Baselines for Structure Learning on Node Regression Task

Repository for training and evaluating simple baselines over a node regression task using different graph structure learning strategies.

Code to reproduce part of the experiments described in "Graph Machine Learning for fast product development from formulation trials", accepted at ECML PKDD 2024.

## Run on your dataset

To train and evaluate all the baselines on your dataset you can run the following command by terminal:
```
python main.py --dataset data --seed seed_value --lr learning_rate --weight_decay weight_decay_value
```
where data is the name of a CSV-formatted file in the data folder.

You can run the baselines over 5 different seeds, obtaining the average and standard deviation performances, by running:
```
python main.py --dataset data --multiple_seeds
```

## Baselines description

The repository contains five different baselines. Specifically, following the literature on structure learning, we choose an MLP that processes the tabular data as is, GCN and GAT using the KNN graph construction on the tabular data, and DGCNN, where the graph is dynamically constructed using nearest neighbors in the feature space and learned end-to-end with the supervised task.  In addition, we consider a GraphTransformer (GT) baseline, where the graph structure is learned using an attention mechanism over the complete graph.

