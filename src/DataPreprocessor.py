import pandas as pd


class DataProcessor:
    def __init__(self):
        self.median_total_charges = None

    def fit(self, df):
        self.median_total_charges = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        ).median()

    def transform(self, df):
        df = df.copy()

        # Change the datatype from str to object
        str_cols = df.select_dtypes(include='str').columns
        df[str_cols] = df[str_cols].astype(object)

        # Change the Total Charges feature datatype to numeric fill the null values with median
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"], errors="coerce"
        )
        df["Total Charges"].fillna(self.median_total_charges, inplace=True)

        # Drop the irrelevant column
        if ["Count", "Country", "State", "CustomerID", "Lat Long", "Latitude", "Longitude", "Zip Code"] in df.columns:
            df = df.drop(
                columns=["Count", "Country", "State", "CustomerID", "Lat Long", "Latitude", "Longitude", "Zip Code"])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
