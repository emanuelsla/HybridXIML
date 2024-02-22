import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

y_name = 'diabetes'
nominal_column_names = [y_name]
metric_column_names = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi',
                       'diabetespedigreefunction', 'age']
ordinal_column_names = []
ordinal_encoding = {}

char_column_names = nominal_column_names


def preprocess_diabetes_data(path: str, test_ratio: float, seed=None) -> ((pd.DataFrame, pd.DataFrame), (dict, dict)):
    """
    function to preprocess diabetes data set

    :param path: path to csv
    :param test_ratio: ratio of test data
    :param seed: random seed
    :return: ((df_train, df_test), (label encoder dict, scaler dict))
    """

    df = pd.read_csv(path, index_col=False)

    df = df.rename(columns=lambda x: x.lower())
    df['diabetes'] = ['yes' if y == 1 else 'no' for y in list(df['outcome'])]
    df = df.drop(columns=['outcome'])

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
        preprocess_diabetes_data('data/diabetes.csv', test_ratio=0.3)

    print(df_train)
