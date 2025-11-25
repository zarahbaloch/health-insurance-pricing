import os
import pandas as pd
from src.data import load_insurance_data, basic_info


def test_load_insurance_data_exists():
    path = "data/raw/insurance.csv"
    assert os.path.exists(path)


def test_load_insurance_data_shape():
    path = "data/raw/insurance.csv"
    df = load_insurance_data(path)
    
    assert df.shape[0] > 0
    assert df.shape[1] == 7
    assert "charges" in df.columns


def test_load_insurance_data_columns():
    path = "data/raw/insurance.csv"
    df = load_insurance_data(path)
    
    expected_cols = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    for col in expected_cols:
        assert col in df.columns


def test_basic_info():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "charges": [1000, 2000, 3000]
    })
    
    info = basic_info(df)
    
    assert "n_rows" in info
    assert "n_cols" in info
    assert "missing_values" in info
    assert "duplicates" in info
    assert info["n_rows"] == 3
    assert info["n_cols"] == 2
