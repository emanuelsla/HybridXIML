import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

y_name = 'diagnosis'
ordinal_column_names = []
ordinal_encoding = {}
nominal_column_names = [y_name]
metric_column_names = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                       "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                       "fractal_dimension_mean"]

drop_cols = ["id", "radius_se", "texture_se", "perimeter_se", "area_se",
             "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
             "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
             "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
             "symmetry_worst", "fractal_dimension_worst"]


def preprocess_diagnostic_data(path: str,
                                          test_ratio: float, seed=None) -> ((pd.DataFrame, pd.DataFrame), (dict, dict)):
    """
    function to preprocess diagnostic data set

    :param path: path to csv
    :param test_ratio: ratio of test data
    :param seed: random seed
    :return: ((df_train, df_test), (label encoder dict, scaler dict))
    """

    df = pd.read_csv(path)

    df.columns = df.columns.str.replace('"', '')
    df = df.dropna()
    for col in drop_cols:
        df = df.drop(col, axis=1)

    label_transformers = {}
    for col in nominal_column_names:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_transformers[col] = le

    metric_transformers = {}
    for col in metric_column_names:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(np.asarray(df[col]).reshape((-1, 1))).flatten()
        metric_transformers[col] = scaler

    df_train, df_test = train_test_split(df, test_size=int(df.shape[0] * test_ratio), random_state=seed)

    return df_train, df_test, label_transformers, metric_transformers


if __name__ == '__main__':

    df_train, df_test, label_transformers, metric_transformers = \
        preprocess_diagnostic_data('data/diagnostic.csv', test_ratio=0.3)

    print(df_train)
