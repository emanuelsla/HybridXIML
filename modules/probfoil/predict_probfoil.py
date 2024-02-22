import pandas as pd


def predict_probfoil(theory: dict, df_test: pd.DataFrame, threshold=0.5) -> list:
    """
    multilabel prediction from probfoil theory
    """

    def predict_with_theory(theory_df_list: list, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        method to predict with single theory
        """

        df_prediction_theory = pd.DataFrame()
        for n, rule_df in enumerate(theory_df_list):
            df_prediction_rule = pd.DataFrame()

            for i in list(rule_df.index):
                column = str(rule_df.loc[i, 'negation']).lower() + '_' + str(rule_df.loc[i, 'predicate']) \
                         + '_' + str(rule_df.loc[i, 'value'])
                if rule_df.loc[i, 'negation']:
                    df_prediction_rule[column] = df_test.loc[:, rule_df.loc[i, 'predicate']] != rule_df.loc[i, 'value']
                else:
                    df_prediction_rule[column] = df_test.loc[:, rule_df.loc[i, 'predicate']] == rule_df.loc[i, 'value']

            all_true = df_prediction_rule.all(axis=1)
            all_true = [rule_df.loc[0, 'probability'] if pred == True else False for pred in all_true]
            df_prediction_theory['rule_' + str(n)] = all_true

        df_prediction_theory['final'] = df_prediction_theory.sum(axis=1)
        df_prediction_theory.index = df_test.index

        return df_prediction_theory

    # iterate over theories for different labels
    y_pred = [None] * len(df_test)
    for label, theory_df_list in theory.items():
        df_pred = predict_with_theory(theory[label], df_test)
        for i, y in enumerate(df_pred['final']):
            if y_pred[i] is None:
                if y > threshold:
                    y_pred[i] = label

    if None in y_pred:
        print('predict_probfoil: Warning: Incomplete theory.')
        print()

    return y_pred

