# import modules
import sys
from pathlib import Path

# import libraries
import pandas as pd
from pandas import DataFrame, Series

from src.config import load_config
from src.exception import CustomException
from src.logger import logging

# load the config
config = load_config()

# locate the root directory
ROOT_DIR = Path(__file__).parents[2]


# =================================================================================================
# --- 1. Separate input and output feature ---
# =================================================================================================

def _input_output_split(df: pd.DataFrame):
    """
    Helper to split data into input and output features
    :param df: pandas.DataFrame
    :return: the input and output dataframe
    """
    try:
        X = df.drop("Churn Value", axis=1)
        y = df["Churn Value"]
        logging.info(f"Dataframe split into input and output feature: {X.shape}, {y.shape}")

        return X, y

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 2. Drop redundant features ---
# =================================================================================================

def _drop_redundant_features(X: pd.DataFrame):
    """
    Helper function to drop redundant features for model training
    :param X: pd dataframe
    :return: pd dataframe
    """
    try:
        X = X.drop(columns=[
            "Churn Label",
            "Churn Reason",
            'Latitude',
            'Longitude',
            'Lat Long',
            'Zip Code',
            'Churn Score',
        ])

        logging.info(f"Dropped redundant features: {X.shape}")
        return X

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 3. Replace the services columns with number of services ---
# =================================================================================================

def _refactor_service_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to create a new feature from how many services the customers have.
    :param X: pd.DataFrame
    :return: pd.DataFrame
    """
    # get the services feature
    services_feature = [
        'Phone Service',
        'Multiple Lines',
        'Internet Service',
        'Online Security',
        'Online Backup',
        'Device Protection',
        'Tech Support',
        'Streaming TV',
        'Streaming Movies'
    ]

    # Create a new service count feature
    X["Service_count"] = (X[services_feature] != "No").sum(axis=1)

    # drop this redundant columns
    X = X.drop(columns=services_feature)

    logging.info(f"Refactored service feature: {X.shape}")

    return X


# =================================================================================================
# --- 4. Refactor tenure month column into categories ---
# =================================================================================================

def _refactor_tenure_month_feature(X: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to refactor tenure month feature into categories
    :param X: pd dataframe
    :return: pd dataframe
    """

    X["customer_loyalty"] = pd.cut(x=X["Tenure Months"],
                                   bins=[0, 1, 2, 10, 29, 72],
                                   labels=["new", "seeker", "floater", "loyal", "extra_loyal"]
                                   )

    X = X.drop(columns=["Tenure Months"])

    logging.info(f"Refactored tenure month feature: {X.shape}")

    return X


# =================================================================================================
# --- 5. Combine all the helpers to perform feature engineering ---
# =================================================================================================

def feature_engineering(
        df: pd.DataFrame
) -> tuple[DataFrame, Series]:
    """
    Combine all the helpers to perform feature engineering
    :param df: pandas DataFrame
    :return: Input and output features
    """

    X, y = _input_output_split(df=df)

    X = _drop_redundant_features(X=X)

    X = _refactor_service_features(X=X)

    X = _refactor_tenure_month_feature(X=X)

    logging.info(f"Feature engineering done: {X.shape}, {y.shape}")

    return X, y
