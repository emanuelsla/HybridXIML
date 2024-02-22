import pandas as pd
import numpy as np


def generate_random_df(df: pd.DataFrame, y_name: str, size: int, seed=None) -> pd.DataFrame:
    if seed:
        np.random.seed(seed)
    columns = list(df.columns)
    random_df_dict = {}
    for col in columns:
        if col != y_name:
            pop = list(df[col])
        else:
            pop = np.unique(df[y_name])
        sample = np.random.choice(pop, size, replace=True)
        random_df_dict[col] = list(sample)
    random_df = pd.DataFrame.from_dict(random_df_dict)
    return random_df


def insert_spurious_correlations(df, y_name, nr_spurious=1, seed=42):
    if seed:
        np.random.seed(seed)

    for n in range(nr_spurious):

        spurious_name = 'spurious_' + str(n + 1)

        df[spurious_name] = df[y_name]

        for i in range(len(df)):
            u = np.random.uniform()
            if u > 0.9:
                labels = list(df[y_name].values)
                labels.remove(df[y_name].iloc[i])
                df.iloc[i, df.columns.get_loc(y_name)] = np.random.choice(labels)

    return df
