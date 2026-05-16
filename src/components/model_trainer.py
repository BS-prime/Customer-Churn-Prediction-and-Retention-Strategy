# import libraries
import sys
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.config import load_config
from src.exception import CustomException
from src.logger import logging


def model_trainer(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config_file=load_config()
)-> list:
    try:
        # map config to actual sklearn classes
        algo_map = {
            "LogisticRegression": LogisticRegression,
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": xgb.XGBClassifier
        }

        models: list = []

        for model_config in config_file["models"].values():
            model_type = model_config["type"]

            base_model = algo_map[model_type](random_state=config_file["training"]["random_state"])

            logging.info(f"Training model: {model_type}")

            # initializing GridSearchCV
            grid = GridSearchCV(
                base_model,
                param_grid=model_config["params"],
                cv=4,
                verbose=3,
                scoring="roc_auc",
            )

            # train the model
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            logging.info(f"Best model: {model_type}")

            # add the models
            models.append(best_model)

        return models

    except Exception as e:
        raise CustomException(e, sys)