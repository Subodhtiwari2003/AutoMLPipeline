import pandas as pd
def load_data(path):
    return pd.read_csv(path)
def quick_profile(df, target_col=None):
    print('Shape:', df.shape)
    print('\nDtypes:\n', df.dtypes)
    print('\nMissing values:\n', df.isna().sum().sort_values(ascending=False).head(20))
    if target_col and target_col in df.columns:
        print('\nTarget distribution:\n', df[target_col].value_counts(normalize=True).head())
