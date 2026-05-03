from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib

class ChurnModel:

    def __init__(self):
        self.pipeline = None

    def build_pipeline(self, X):

        cat_cols = X.select_dtypes(include="object").columns
        num_cols = X.select_dtypes(exclude="object").columns

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ])

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )

        self.pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

    def fit(self, X, y):
        self.build_pipeline(X)
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)