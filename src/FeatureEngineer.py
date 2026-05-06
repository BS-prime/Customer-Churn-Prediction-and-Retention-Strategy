import pandas
import pandas as pd


class FeatureEngineer:
    def transform(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # -------------------------------------------------------------------------
        # --- 0. Make a shallow copy ---
        # -------------------------------------------------------------------------
        df = df.copy()

        # -------------------------------------------------------------------------
        # --- 1. Binning of the Tenure Month column ---
        # -------------------------------------------------------------------------

        df["tenure_group"] = pd.cut(
            df["Tenure Month"],
            bins=[0, 12, 36, 60, 100],
            labels=["new", "mid", "loyal", "very_loyal"],
        )

        df.drop(columns=["Tenure Month"], axis=1, inplace=True)

        # -------------------------------------------------------------------------
        # --- 2. Convert services into service count ---
        # -------------------------------------------------------------------------

        service_cols = [
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
        ]

        df["service_count"] = (df[service_cols] != "No").sum(axis=1)

        df.drop(
            [
                "Phone Service",
                "Multiple Lines",
                "Internet Service",
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies",
            ],
            axis=1,
            inplace=True,
        ) # drop the redundant columns

        # -------------------------------------------------------------------------
        # --- 3. Handle cardinal feature (City) ---
        # -------------------------------------------------------------------------

        city_stats = df.groupby("City")["Churn Value"].agg(["mean", "count"])

        reliable_cities = city_stats[city_stats["count"] >= 10]
        top_churn_cities = (
            reliable_cities.sort_values(by="mean", ascending=False).head(10).index
        )
        df["city_hotspot"] = df["City"].apply(
            lambda x: x if x in top_churn_cities else "Other"
        )

        # one hot encode the new column
        df = pd.get_dummies(df, columns=["city_hotspot"], prefix="hotspot")

        df.drop(columns=["City"], axis=1, inplace=True)

        # -------------------------------------------------------------------------
        # --- 4. One hot encode the columns ---
        # -------------------------------------------------------------------------

        obj_cols = df.select_dtypes(include="object").columns.to_list()

        df = pd.get_dummies(data=df, columns=obj_cols, drop_first=True)

        return df
