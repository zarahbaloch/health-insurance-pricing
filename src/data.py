import pandas as pd

def load_insurance_data(path):
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame):
    info = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'column_types': df.dtypes.astype(str).to_dict(),
        'n_unique': df.nunique().to_dict(),
        'basic_stats': df.describe().to_dict(),
        'value_counts_sample': {
            col: df[col].value_counts().head(5).to_dict()
            for col in df.columns
            if df[col].dtype == 'object'
        }
    }
    return info