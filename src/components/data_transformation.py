# import modules
import sys
from pathlib import Path
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

# import libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Locate the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

@dataclass
class DataTransformationConfig:
    preprocessor: Path = ROOT_DIR / "artifacts" / "preprocessor.pkl"

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_obj(self):
        try:
            