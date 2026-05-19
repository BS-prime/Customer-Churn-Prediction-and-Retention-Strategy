# import libraries
import pandas as pd

# import modules
from logger import logging


def _handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to handle missing values in Total Charges feature with
    0.0
    :param dataframe: pandas.DataFrame
    :return: pandas dataframe with missing values
    """
    dataframe["Total Charges"] = (dataframe["Total Charges"]
                                  .replace(" ", 0.0)
                                  ).infer_objects(copy=False)

    logging.info("Replace null values with np.nans")

    return dataframe


def _fixing_datatype(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fixing datatypes in Total Charges feature
    :param dataframe:
    :return: pandas dataframe with fixed datatypes
    """

    # Change the data type of the Total Charges feature
    dataframe["Total Charges"] = dataframe["Total Charges"].astype(float)

    logging.info("Fixing datatypes in dataframe")

    return dataframe


def _drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns in dataframe
    :param dataframe:
    :return:
    """

    # Drop the zero variance columns in the dataframe
    dataframe = dataframe.drop(
        columns=[
            "Count",
            "Country",
            "State",
            "CustomerID"
        ]
    )

    logging.info(
        f"Drop columns zero variance columns from dataframe: {dataframe.shape}"
    )

    return dataframe


def data_cleaner(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe with helper functions
    :param dataframe:
    :return: the cleaned dataframe
    """

    dataframe = _handle_missing_values(dataframe)
    dataframe = _fixing_datatype(dataframe)
    dataframe = _drop_columns(dataframe)

    logging.info("Data cleaning completed")

    return dataframe
