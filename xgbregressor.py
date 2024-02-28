import xgboost as xgb
import numpy as np
from torch_geometic.data import Data
from sklearn.model_selection import GridSearchCV

class XGBTorch():
    """
        Pytorch wrap for XGBoost Regressor. NB: it is not a torch nn module!!
    """

    def __init__(self):
        self.xgb_model = xgb.XGBRegressor()
        self.best_model = None

    def train_grid_search(self, data: Data):
        # Define the parameter grid
        param_grid = {
            'learning_rate': [0.01],
            'max_depth': [3, 4, 5],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }

        grid_search = GridSearchCV(estimator=self.xgb_model, param_grid=param_grid, cv=5, n_jobs=-1)

        x = data.x.detach().numpy()
        y = data.x.detach().numpy()

        grid_search.fit(x, y)

        # Print the best hyperparameters
        print("Best hyperparameters:", grid_search.best_params_)

        self.best_model = grid_search.best_estimator_

    def test(self, data: Data):
        if self.best_model is None:
            raise Exception('train must be called first')
        x = data.x.detach().numpy()
        y_hat = self.best_model.predict(x)
        return y_hat
