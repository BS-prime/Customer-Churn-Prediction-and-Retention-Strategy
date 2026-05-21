# Customer Churn Prediction and Retention Strategy

An end-to-end machine learning project for predicting customer churn and translating churn risk into an actionable retention strategy. The project uses the Telco customer churn dataset, builds a reusable training pipeline, compares multiple classifiers, optimizes the decision threshold with business costs, and estimates whether a retention offer is profitable for each customer.

## Project Overview

Customer churn is one of the most important problems for subscription businesses. A churn model is useful only when it helps the business decide which customers should receive retention action. This project focuses on both parts:

- Predicting the probability that a customer will churn.
- Identifying high-risk customers using a cost-aware threshold.
- Estimating retention profit from an intervention offer.
- Saving reusable model, preprocessing, metrics, and threshold artifacts.

The pipeline is built in a modular way so each stage can be inspected, tested, or replaced independently.

## Business Problem

The goal is to help a telecom company reduce customer churn by identifying customers who are likely to leave and prioritizing retention offers where the expected value is positive.

Instead of treating churn prediction as a purely technical classification task, the project adds business-aware decision logic:

- False negatives are expensive because missed churners may leave.
- False positives are less expensive because they mainly represent unnecessary retention offer cost.
- The final threshold is selected by minimizing a custom cost function.
- Retention profit is estimated using churn probability, offer cost, monthly revenue, retention success rate, and expected retained months.

## Dataset

The project uses the Telco customer churn dataset stored at:

```text
data/raw/Telco_customer_churn.xlsx
```

The raw dataset includes customer demographics, account information, subscribed services, billing details, location fields, churn labels, churn scores, and churn reasons.

## Workflow

The training workflow is implemented in `src/pipeline/train_pipeline.py` and follows these stages:

1. Data ingestion
   - Loads the raw Excel dataset from the configured data path.

2. Data cleaning
   - Handles blank values in `Total Charges`.
   - Converts `Total Charges` to a numeric type.
   - Drops zero-variance or identifier columns such as `Count`, `Country`, `State`, and `CustomerID`.

3. Feature engineering
   - Splits the target variable `Churn Value` from the input features.
   - Drops leakage or redundant fields such as `Churn Label`, `Churn Reason`, `Churn Score`, latitude/longitude, and zip/location helper fields.
   - Converts service-related columns into a single `Service_count` feature.
   - Converts `Tenure Months` into a categorical customer loyalty feature.

4. Data preprocessing
   - Performs a stratified train-test split.
   - Applies median imputation and scaling to numerical features.
   - Applies most-frequent imputation and one-hot encoding to categorical features.
   - Applies target encoding to high-cardinality city data.
   - Saves transformed train and test feature arrays.

5. Model training
   - Trains multiple models using `GridSearchCV`.
   - Uses ROC-AUC as the grid-search scoring metric.
   - Model configurations are managed in `configs/config.yaml`.

6. Model evaluation
   - Evaluates trained models on the test set.
   - Saves metrics including ROC-AUC, accuracy, precision, recall, F1 score, and balanced accuracy.
   - Selects the best model using recall and F1-oriented logic.

7. Threshold optimization
   - Searches thresholds from 0.01 to 0.99.
   - Minimizes business cost using false-positive and false-negative costs.
   - Saves the best threshold and cost information.

8. Retention strategy
   - Converts churn probability into expected retention profit.
   - Helps decide which customers should receive offers.

## Models Compared

The project currently compares:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Hyperparameters for each model are configured in:

```text
configs/config.yaml
```

## Current Model Results

The latest saved evaluation metrics are available at:

```text
artifacts/metrics/metrics.json
```

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8304 | 0.7445 | 0.5127 | 0.7567 | 0.6112 | 0.7484 |
| Random Forest | 0.7929 | 0.7722 | 0.6091 | 0.3957 | 0.4797 | 0.6520 |
| XGBoost | 0.7679 | 0.5607 | 0.3646 | 0.8824 | 0.5160 | 0.6634 |

The saved threshold optimization result is:

```json
{
  "best_threshold": 0.2,
  "min_cost": 696.0,
  "cost_fp": 1.0,
  "cost_fn": 10.0
}
```

This means the project currently favors catching more churners, because missing a churner is configured as 10 times more costly than incorrectly targeting a non-churner.

## Retention Profit Logic

The retention strategy is implemented in:

```text
src/retention_strategy/retention.py
```

The expected profit formula is:

```text
expected_gain = churn_probability * retention_success * monthly_revenue * retained_months
profit = expected_gain - offer_cost
```

Default assumptions:

- Offer cost: 20
- Monthly revenue: 70
- Retention success rate: 30%
- Retained period: 12 months

A customer can be targeted when the expected retention profit is positive.

## Project Structure

```text
.
|-- artifacts/
|   |-- features/              # Transformed train and test features
|   |-- explainability/        # Saved SHAP summary plot
|   |-- metrics/               # Saved model evaluation metrics
|   |-- preprocessor/          # Saved preprocessing object
|   `-- threshold/             # Saved optimal threshold
|-- configs/
|   `-- config.yaml            # Data paths, model params, and output paths
|-- data/
|   `-- raw/                   # Raw and cleaned datasets
|-- logs/                      # Pipeline logs
|-- models/
|   `-- best_model.pkl         # Saved best model
|-- notebooks/
|   |-- eda.ipynb
|   |-- preprocessing.ipynb
|   |-- feature_engineering.ipynb
|   `-- model_training_and_evaluation.ipynb
|-- src/
|   |-- api/                   # FastAPI app and request/response schemas
|   |-- components/            # Modular ML pipeline components
|   |-- pipeline/              # Training and reusable prediction logic
|   |-- retention_strategy/    # Churn probability and retention profit logic
|   |-- ui/                    # Streamlit customer scoring interface
|   |-- config.py
|   |-- exception.py
|   |-- logger.py
|   `-- utils.py
|-- main.py
|-- pyproject.toml
`-- README.md
```

## Installation

This project uses Python 3.12 or higher.

Clone the repository and install the dependencies:

```bash
pip install -e .
```

If you use `uv`, you can install from the lock file:

```bash
uv sync
```

## How to Run the Training Pipeline

Run the training pipeline from the project root:

```bash
python -m src.pipeline.train_pipeline
```

The pipeline will create or update:

- `artifacts/features/train.csv`
- `artifacts/features/test.csv`
- `artifacts/preprocessor/preprocessor.pkl`
- `models/best_model.pkl`
- `artifacts/metrics/metrics.json`
- `artifacts/threshold/threshold.json`
- `artifacts/explainability/shap_summary.png`

## How to Run the FastAPI Prediction Service

The project includes a FastAPI app in:

```text
src/api/app.py
```

Start the API from the project root:

```bash
uvicorn src.api.app:app --reload
```

The service exposes:

- `GET /health`: checks whether the API is running.
- `POST /predict`: returns churn probability, churn decision, expected retention profit, and whether the customer should be targeted.

The reusable prediction logic lives in:

```text
src/pipeline/predict_pipeline.py
```

Example request:

```json
{
  "MonthlyCharges": 100,
  "TotalCharges": 1000,
  "CLTV": 4226,
  "ServiceCount": 1,
  "City": "Los Angeles",
  "Gender": "Male",
  "SeniorCitizen": "No",
  "Partner": "No",
  "Dependents": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "CustomerLoyalty": "new"
}
```

Example response:

```json
{
  "model_version": "1.0.0",
  "churn_probability": 0.7345,
  "will_churn": true,
  "profit": 165.49,
  "should_target": true
}
```

## How to Run the Streamlit App

The project includes a Streamlit app in:

```text
src/ui/streamlit_app.py
```

Start the app from the project root:

```bash
streamlit run src/ui/streamlit_app.py
```

The app loads the saved model artifacts, accepts customer details, and returns churn probability, churn decision, expected retention profit, and targeting recommendation.

## How to Run Tests

Run the focused test suite with the development extra:

```bash
uv run --extra dev pytest
```

You can also run a compile check with:

```bash
py -m compileall src tests
```

## Configuration

Most project settings are controlled through:

```text
configs/config.yaml
```

This file includes:

- Raw data path
- Train-test split size
- Random state
- Model hyperparameter grids
- Output paths for features, model, preprocessor, threshold, and metrics

## Notebooks

The notebooks document the experimentation process:

- `eda.ipynb`: exploratory data analysis.
- `preprocessing.ipynb`: preprocessing experiments.
- `feature_engineering.ipynb`: feature creation and transformation ideas.
- `model_training_and_evaluation.ipynb`: model training and comparison.

## Key Highlights

- Modular production-style pipeline.
- Config-driven model training and artifact paths.
- Multiple model comparison with hyperparameter tuning.
- Stratified split for churn imbalance.
- Separate preprocessing for numeric, categorical, and high-cardinality features.
- Cost-sensitive threshold optimization.
- Retention profit calculation for business decision-making.
- SHAP summary plot for model explainability.
- FastAPI prediction service for real-time churn scoring.
- Streamlit interface for customer scoring.
- Saved artifacts for reproducibility and reuse.
- Logs for pipeline traceability.

## Future Improvements

- Add automated tests for each pipeline component.
- Add detailed SHAP dependence plots for deeper model explainability.
- Improve best-model selection with a clearer cost-sensitive objective.
- Track experiments with MLflow or another experiment registry.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- category-encoders
- PyYAML
- dill / joblib
- Jupyter notebooks
