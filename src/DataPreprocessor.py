import pandas
import pandas as pd


class DataProcessor:
    def __init__(self):
        self.median_total_charges = None

    def fit(self, df):

        self.median_total_charges = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        ).median()

    def transform(self, df):

        # ------------------------------------------------------------------------------
        # --- 0. Make a shallow copy ---
        # ------------------------------------------------------------------------------

        df = df.copy()

        # ------------------------------------------------------------------------------
        # --- 1. change datatype(object to str) ---
        # ------------------------------------------------------------------------------

        str_cols = df.select_dtypes(include='str').columns
        df[str_cols] = df[str_cols].astype(object)

        # ------------------------------------------------------------------------------
        # --- 2. replace with median value ---
        # ------------------------------------------------------------------------------

        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"], errors="coerce"
        )
        df["Total Charges"].fillna(self.median_total_charges, inplace=True)

        # ------------------------------------------------------------------------------
        # --- 3. Drop redundant columns ---
        # ------------------------------------------------------------------------------

        df = df.drop(
            columns=["Count", "Country", "State", "CustomerID", "Lat Long", "Latitude", "Longitude", "Zip Code"])

        return df

    def fit_transform(self, df: pandas.DataFrame) -> pandas.DataFrame:
        self.fit(df)

        return self.transform(df)
