# RiskRateML: Predictive Modeling for Insurance Charges

This project predicts health insurance charges using the Medical Cost Personal Dataset
(Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance). The repo is structured as a full pipeline with modular code, unit tests, and a final modeling notebook.

**Notebook Overview** 
- Load Data
- Exploratory Data Analysis
- Feature Engingeering
- Train/ Test Split
- Scaling
- Modeling (Linear Regression, Random Forest Regressor)
- Feature Importance

**Source Code (src/)**
- src/data.py (Loads the dataset and provides basic EDA utilities)
- src/features.py (Encodes categorical variables and splits the data into features/target) 
- src/ models.py (Train/test split helper, scaling functions, and wrappers for LR + RF training) 
- src/utils.py (Metric calculations and clean metric-printing functions)

  **Test Suite (tests/)**
Each part of the pipeline is validated using pytest

**Model Performance Summary** 
1) Linear Regression
    - MAE: 4181 (predicts charges with an average error of $4181) 
    - RMSE: 5796 (typical deviation is around $5796)
    - R2: 0.7836 (explains 78.36% of the data)

Predictive performance is reasonable but limited by non-linear relationships in the data.
  
2) Random Forest (Stronger Model) 
    - MAE: 2558 (predicts charges with an average error of $2558)
    - RMSE: 4576 (typical deviation is around $4576)
    - R2: 0.8651 (explains 86.51% of the data

Stronger model overall, captures interactions and non-linear structure in charges.
  
