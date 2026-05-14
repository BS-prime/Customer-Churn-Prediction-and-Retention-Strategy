# import modules
import sys
from pathlib import Path
from typing import Any

from numpy import dtype, ndarray
from pandas import DataFrame, Series

from src.exception import CustomException
from src.logger import logging
from src.config import load_config

# import libraries
import pandas as pd
import category_encoders as ce
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import save_object

# load the config
config = load_config()


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
        logging.info(f"Dataframe split into input and output feature: {df.shape}, {y.shape}")

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
            "Churn Reason"
        ])

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

    return X


# =================================================================================================
# --- 5. Perform train test split ---
# =================================================================================================

def _train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = config["data"]["test_size"],
        random_state: int = config["data"]["random_state"]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Helper function to split data into train and test
    :param X: pandas.DataFrame
    :param y: pandas Series
    :param test_size: float
    :param random_state: int
    :return: X_train, X_test, y_train, y_test
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            stratify=y,
                                                            test_size=test_size,
                                                            random_state=random_state
                                                            )
        logging.info(f"Dataframe split into train and test: "
                     f"{X_train.shape}, "
                     f"{X_test.shape}, "
                     f"{y_train.shape}, "
                     f"{y_test.shape}"
                     )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 6. Select features based on datatype ---
# =================================================================================================

def _select_datatypes_columns(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Helper to select columns of numerical and categorical data types from a pd dataframe
    :param X: pd dataframe
    :return: categorical, cardinal, and numerical column lists
    """
    try:
        # select the features according to datatype
        cat_columns = X.select_dtypes(include="object").columns.drop(["City"]).tolist()
        cardinal_columns = ["City"]
        num_columns = X.select_dtypes(include="number").columns.tolist()

        logging.info(f"Columns of numerical data: {num_columns}")
        logging.info(f"Columns of cardinal data: {cardinal_columns}")
        logging.info(f"Columns of categorical data: {cat_columns}")

        return cat_columns, cardinal_columns, num_columns

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 7. Create pipelines for different datatypes ---
# =================================================================================================

def _create_pipelines() -> tuple[Pipeline, Pipeline, Pipeline]:
    """
    Helper function to create pipeline objects of numerical, categorical, and cardinal data
    :return: num_pipeline, cat_pipeline, card_pipeline
    """
    try:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        logging.info(f"Steps of numerical pipeline: {num_pipeline.steps}")

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),  # Added handle_unknown for safety
        ])

        logging.info(f"Steps of categorical pipeline: {cat_pipeline.steps}")

        card_pipeline = Pipeline([
            ("target_encoder", ce.TargetEncoder(smoothing=10))
        ])

        logging.info(f"Steps of cardinal pipeline: {card_pipeline.steps}")

        return num_pipeline, cat_pipeline, card_pipeline

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 8. Create preprocessor out of those pipelines ---
# =================================================================================================

def _create_preprocessor(
        num_pipeline: Pipeline,
        cat_pipeline: Pipeline,
        card_pipeline: Pipeline,
        cat_columns: list[str],
        num_columns: list[str],
        card_columns: list[str],
) -> ColumnTransformer:
    """
    Helper to combine all the pipelines to create a preprocessor
    :param num_pipeline: Pipeline
    :param cat_pipeline: Pipeline
    :param card_pipeline: Pipeline
    :return: Returns a ColumnTransformer object
    """
    try:
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_columns),
            ("cat", cat_pipeline, cat_columns),
            ("card", card_pipeline, card_columns),
        ])

        logging.info("Preprocessor initialized}")

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 9. Using preprocessor to transform the data ---
# =================================================================================================

def _initiate_data_transformation(
        preprocessor_obj: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        config_data=config["output"],
) -> Path:
    """
    Helper to transform the data, save features and save the preprocessor object
    :param preprocessor_obj: ColumnTransformer object
    :param X_train: pandas DataFrame
    :param y_train: pandas Series
    :param X_test: pandas DataFrame
    :param y_test: pandas Series
    :param config_data: Path
    :return: transformed train array, transformed test array, and file path to preprocessor object
    """
    try:
        # Fit preprocessor to the training data USING y_train for the TargetEncoder
        input_feature_train_arr = preprocessor_obj.fit_transform(X_train, y_train)

        # Transform the test data
        input_feature_test_arr = preprocessor_obj.transform(X_test)

        # Concatenate transformed features with their respective target variables
        train_arr = np.c_[input_feature_train_arr, np.array(y_train)]
        test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

        # Save preprocessor
        save_object(
            filepath=config_data["preprocessor_path"],
            obj=preprocessor_obj
        )

        # save feature engineered data
        train_arr.to_csv(config_data["train_feature_path"], index=False, header=True)
        test_arr.to_csv(config_data["test_feature_path"], index=False, header=True)

        return (
            config_data["preprocessor_path"]
        )

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 10. Combine all the helpers to perform feature engineering ---
# =================================================================================================

def feature_engineering(
        df: pd.DataFrame
) -> tuple[DataFrame, DataFrame, Series, Series, Path]:
    """
    Combine all the helpers to perform feature engineering
    :param df: pandas DataFrame
    :return: X_train, X_test, y_train, y_test, and Object file path
    """

    X, y = _input_output_split(df=df)

    X = _drop_redundant_features(X=X)

    X = _refactor_service_features(X=X)

    X = _refactor_tenure_month_feature(X=X)

    X_train, X_test, y_train, y_test = _train_test_split(X, y)

    cat_columns, num_columns, card_columns = _select_datatypes_columns(X=X_train)

    num_pipeline, cat_pipeline, card_pipeline = _create_pipelines()

    preprocessor_obj = _create_preprocessor(num_pipeline=num_pipeline,
                                            cat_pipeline=cat_pipeline,
                                            card_pipeline=card_pipeline,
                                            card_columns=card_columns,
                                            num_columns=num_columns,
                                            cat_columns=cat_columns
                                            )

    preprocessor_file_path = _initiate_data_transformation(preprocessor_obj=preprocessor_obj,
                                                           X_train=X_train,
                                                           y_train=y_train,
                                                           X_test=X_test,
                                                           y_test=y_test,
                                                           )

    return X_train, X_test, y_train, y_test, preprocessor_file_path
