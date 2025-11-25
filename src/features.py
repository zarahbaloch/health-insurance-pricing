import pandas as pd

def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    cat_cols = ["sex", "smoker", "region"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


def prepare_features_target(df: pd.DataFrame, target_col: str = "charges"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
