# Baselines for Graph Structure Learning on Node Regression Task

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

## Cite
If you use the code of this repository for your project or you find the work interesting, please cite the following work: 

Dileo, M., Olmeda, R., Pindaro, M., Zignani, M. (2024). Graph Machine Learning for Fast Product Development from Formulation Trials. In: Bifet, A., Krilavičius, T., Miliou, I., Nowaczyk, S. (eds) Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track. ECML PKDD 2024. Lecture Notes in Computer Science(), vol 14949. Springer, Cham. https://doi.org/10.1007/978-3-031-70378-2_19

```bibtex
@InProceedings{DileoIntellico2024,
author="Dileo, Manuel
and Olmeda, Raffaele
and Pindaro, Margherita
and Zignani, Matteo",
editor="Bifet, Albert
and Krilavi{\v{c}}ius, Tomas
and Miliou, Ioanna
and Nowaczyk, Slawomir",
title="Graph Machine Learning for Fast Product Development from Formulation Trials",
booktitle="Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="303--318",
abstract="Product development is the process of creating and bringing a new or improved product to market. Formulation trials constitute a crucial stage in product development, often involving the exploration of numerous variables and product properties. Traditional methods of formulation trials involve time-consuming experimentation, trial and error, and iterative processes. In recent years, machine learning (ML) has emerged as a promising avenue to streamline this complex journey by enhancing efficiency, innovation, and customization. One of the paramount challenges in ML for product development is the models' lack of interpretability and explainability. This challenge poses significant limitations in gaining user trust, meeting regulatory requirements, and understanding the rationale behind ML-driven decisions. Moreover, formulation trials involve the exploration of relationships and similarities among previous preparations; however, data related to formulation are typically stored in tables and not in a network-like manner. To cope with the above challenges, we propose a general methodology for fast product development leveraging graph ML models, explainability techniques, and powerful data visualization tools. Starting from tabular formulation trials, our model simultaneously learns a latent graph between items and a downstream task, i.e. predicting consumer-appealing properties of a formulation. Subsequently, explainability techniques based on graphs, perturbation, and sensitivity analysis effectively support the R{\&}D department in identifying new recipes for reaching a desired property. We evaluate our model on two datasets derived from a case study based on food design plus a standard benchmark from the healthcare domain. Results show the effectiveness of our model in predicting the outcome of new formulations. Thanks to our solution, the company has drastically reduced the labor-intensive experiments in real laboratories and the waste of materials.",
isbn="978-3-031-70378-2"
}

```

