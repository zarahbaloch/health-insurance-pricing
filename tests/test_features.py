import pandas as pd
from src.features import encode_categoricals, prepare_features_target


def test_encode_categoricals_removes_original_columns():
    df = pd.DataFrame({
        "age": [20, 30],
        "sex": ["male", "female"],
        "bmi": [25.0, 30.0],
        "children": [0, 1],
        "smoker": ["yes", "no"],
        "region": ["southwest", "southeast"],
        "charges": [1000.0, 2000.0]
    })
    
    result = encode_categoricals(df)
    
    assert "sex" not in result.columns
    assert "smoker" not in result.columns
    assert "region" not in result.columns


def test_encode_categoricals_creates_dummies():
    df = pd.DataFrame({
        "age": [20, 30],
        "sex": ["male", "female"],
        "bmi": [25.0, 30.0],
        "children": [0, 1],
        "smoker": ["yes", "no"],
        "region": ["southwest", "southeast"],
        "charges": [1000.0, 2000.0]
    })
    
    result = encode_categoricals(df)
    
    dummy_cols = [c for c in result.columns if any(x in c for x in ["sex_", "smoker_", "region_"])]
    assert len(dummy_cols) > 0


def test_encode_categoricals_preserves_numeric_columns():
    df = pd.DataFrame({
        "age": [20, 30],
        "sex": ["male", "female"],
        "bmi": [25.0, 30.0],
        "children": [0, 1],
        "smoker": ["yes", "no"],
        "region": ["southwest", "southeast"],
        "charges": [1000.0, 2000.0]
    })
    
    result = encode_categoricals(df)
    
    assert "age" in result.columns
    assert "bmi" in result.columns
    assert "children" in result.columns
    assert "charges" in result.columns


def test_prepare_features_target():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "bmi": [25.0, 30.0, 28.0],
        "charges": [1000.0, 2000.0, 3000.0]
    })
    
    X, y = prepare_features_target(df, target_col="charges")
    
    assert "charges" not in X.columns
    assert "age" in X.columns
    assert "bmi" in X.columns
    assert len(y) == 3
    assert y.name == "charges"