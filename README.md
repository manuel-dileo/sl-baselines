# Baselines for Structure Learning + Node regression task

## Battle Plan

- Minimize RMSE, report RMSE and MAE.
- Optimize the proposed model first, reporting the best model configuration and k (#neighbors to consider).
- To be fair, create baselines with #parameters equal or comparable with the proposed one.
- To be fair, conduct a grid-search on multi-head attention for GraphTransformer.
- To be fair, conduct hyperparameter tuning for XGBoost Regressor.
- To be fair, run the best model for each baselines over 5 different random seed, reporting the average and stdev performances.
