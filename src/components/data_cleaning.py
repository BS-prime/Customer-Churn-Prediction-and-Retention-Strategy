# import libraries
import numpy as np
import pandas as pd


def _handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to handle missing values with np.nans
    :param dataframe:
    :return:
    """
    dataframe["Total Charges"] = dataframe["Total Charges"].replace(" ", np.nan)

    return dataframe


def _fixing_datatype(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fixing datatypes in dataframe
    :param dataframe:
    :return:
    """

    # change the str datatype of each values
    str_cols = dataframe.select_dtypes(include="str").columns
    dataframe[str_cols] = dataframe[str_cols].astype(object)

    # Change the data type of the Total Charges feature
    dataframe["Total Charges"] = dataframe["Total Charges"].astype(float)

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

    return dataframe
