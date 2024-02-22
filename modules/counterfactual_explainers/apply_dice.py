import dice_ml
import warnings
import sklearn
import pandas as pd
import numpy as np
from wrapt_timeout_decorator import *


@timeout(1)
def execute_dice(model: sklearn, instance: pd.DataFrame, df_train: pd.DataFrame,
                 metric_cols: list, y_name: str,
                 number_cfs=5, seed=None) -> dice_ml:
    """
    method to generate counterfactuals with dice

    :param model: sklearn model
    :param instance: pandas data frame with instance to explain
    :param df_train: model training data frame
    :param metric_cols: list of metric column names
    :param y_name: name of target

    :param number_cfs: number of counterfactuals to construct
    :param seed: random seed
    :return cfs: dice_ml object
    """

    d = dice_ml.Data(dataframe=df_train, continuous_features=metric_cols, outcome_name=y_name)
    m = dice_ml.Model(model=model, backend='sklearn')
    dice = dice_ml.Dice(d, m, method='random')

    pop = list(np.unique(df_train[y_name]))
    try:
        pop.remove(instance[y_name].iloc[0])
        desired_class = int(np.random.choice(pop))
    except ValueError:
        desired_class = pop[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cfs = dice.generate_counterfactuals(instance.drop(y_name, axis=1), total_CFs=number_cfs,
                                            desired_class=desired_class, random_seed=seed)

    return cfs


def reconstruct_dice_df(cfs, metric_cols: list, df_type='cfs_list') -> pd.DataFrame:
    """
    method to reconstruct data frames from dice explainer

    :param cfs: dice explainer
    :param metric_cols: list of metric columns
    :param df_type: valid params are 'cfs_list' and 'test_data'
    :return df: reconstructed df
    """

    import json
    json_data = json.loads(cfs.to_json())
    data_array = np.asarray(json_data[df_type])[0]
    df = pd.DataFrame(data=data_array,
                      index=range(data_array.shape[0]),
                      columns=json_data['feature_names_including_target'])

    converter = {}
    for metric_col in metric_cols:
        converter[metric_col] = float
    df = df.astype(converter)

    return df


def apply_dice(model: sklearn, instance: pd.DataFrame, df_train: pd.DataFrame,
               metric_cols: list, y_name: str,
               number_cfs=5, seed=None, verbose=False) -> pd.DataFrame:
    """
    wrapper

    method to apply dice and convert output to pandas data frame

    :param model: sklearn model
    :param instance: pandas data frame with instance to explain
    :param df_train: model training data frame
    :param metric_cols: list of metric column names
    :param y_name: name of target
    :param number_cfs: number of counterfactuals to construct
    :param seed: random seed
    :param verbose: if prints are returned or not
    :return df: pandas data frame with counterfactuals
    """

    if verbose:
        print('generate counterfactuals')
    cfs = execute_dice(model, instance, df_train, metric_cols, y_name, number_cfs, seed)
    if verbose:
        print()
    df = reconstruct_dice_df(cfs, metric_cols)

    return df
