import pandas as pd
import numpy as np
from src.features import encode_categoricals
from src.models import split_train_test, scale_features, train_linear_regression, train_random_forest


def test_split_train_test():
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.rand(100))
    
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_scale_features():
    X_train = pd.DataFrame(np.random.rand(100, 3) * 100)
    X_test = pd.DataFrame(np.random.rand(20, 3) * 100)
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.abs(X_train_scaled.mean()) < 1e-10


def test_train_linear_regression():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "sex": ["male", "female", "male", "female"],
        "bmi": [25.0, 30.0, 28.0, 32.0],
        "children": [0, 1, 2, 3],
        "smoker": ["yes", "no", "no", "yes"],
        "region": ["southwest", "southeast", "northwest", "northeast"],
        "charges": [1000.0, 2000.0, 3000.0, 4000.0]
    })
    
    X = df.drop(columns=["charges"])
    y = df["charges"]
    X_encoded = encode_categoricals(X)
    
    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    model = train_linear_regression(X_train_scaled, y_train)
    
    assert hasattr(model, "predict")
    assert hasattr(model, "coef_")


def test_train_random_forest():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "sex": ["male", "female", "male", "female"],
        "bmi": [25.0, 30.0, 28.0, 32.0],
        "children": [0, 1, 2, 3],
        "smoker": ["yes", "no", "no", "yes"],
        "region": ["southwest", "southeast", "northwest", "northeast"],
        "charges": [1000.0, 2000.0, 3000.0, 4000.0]
    })
    
    X = df.drop(columns=["charges"])
    y = df["charges"]
    X_encoded = encode_categoricals(X)
    
    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y)
    
    model = train_random_forest(X_train, y_train, n_estimators=10)
    
    assert hasattr(model, "predict")
    assert hasattr(model, "feature_importances_")