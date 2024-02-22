import re
import subprocess
import pandas as pd
import numpy as np


def prepare_for_problog(df, label_transformers, metric_transformers, ordinal_encoding):
    df_problog = pd.DataFrame()

    for col, le in label_transformers.items():
        df_problog[col] = list(le.inverse_transform(df[col]))

    for col, scaler in metric_transformers.items():
        if len(df) > 1:
            rescaled = list(scaler.inverse_transform(np.reshape(list(df[col]), (-1, 1))))
        else:
            rescaled = list(scaler.inverse_transform(np.reshape(float(df[col]), (1, -1))))
        rescaled = [r[0] for r in rescaled]
        perc = np.percentile(rescaled, [0, 25, 50, 75, 100])
        quartiles = np.searchsorted(perc, rescaled)
        quartiles = [[0, 25, 50, 75, 100][q] for q in quartiles]
        quartiles = ['q_' + str(q) for q in quartiles]
        df_problog[col] = quartiles

    for col, vals in ordinal_encoding.items():
        df_problog[col] = list(df[col].apply(lambda x: vals[x]))

    df_problog.index = df.index

    return df_problog


def autonomous_code(df: pd.DataFrame, target=(str, str)) -> str:
    df_target = pd.DataFrame(df[target[0]])
    df_var = pd.DataFrame(df.drop([target[0]], axis=1))
    variables = list(df_var.columns)

    settings = ''
    settings += 'base(' + target[0] + '_' + target[1] + '(instance)).\n\n'
    settings += 'base(instance(i)).\n\n'
    for var in variables:
        settings += 'base(' + var + '(instance, value)).\n'
        settings += 'mode(' + var + '(+, c)).\n\n'

    settings += 'learn(' + target[0] + '_' + target[1] + '/1).\n\n'

    examples = ''
    for idx in list(df_var.index):
        examples += 'instance(i_' + str(idx) + ').\n'
        for var in variables:
            examples += var + '(i_' + str(idx) + ', ' + str(df_var.loc[idx, var]) + ').\n'
        if str(df_target.loc[idx, target[0]]) == target[1]:
            examples += '1.0::' + target[0] + '_' + target[1] + '(i_' + str(idx) + ').\n'
        else:
            examples += '0.0::' + target[0] + '_' + target[1] + '(i_' + str(idx) + ').\n'
        examples += '\n'

    code = settings + examples

    return code


def execute_shell(command: str, path: str, args_list: []) -> str:
    """
    function to execute scripts in shell,
    see subprocess module for further documentation

    :param command: valid shell command (e.g. 'probfoil')
    :param path: string containing path to file that should be executed
    :param args_list: optional parameter containing further command arguments as strings (e.g., ['-s 42'])
    return result: string containing shell log
    """

    result = subprocess.run([command, path] + args_list, stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')

    return result


def postprocess_probfoil(result: str) -> list:
    re_rule = r'RULE LEARNED: .* \:\- .*'
    re_float = r'\d+\.\d+'
    re_dcol = r':-'

    # define iterator for rules
    theory_iter = re.finditer(re_rule, result)

    # define output space
    theory_df_list = []

    # iterate over rules in theory
    for rule in theory_iter:

        # define rule df
        rule_df = pd.DataFrame(columns=['probability', 'negation', 'predicate', 'value'])

        # extract rule
        rule = rule.group()
        rule = rule.replace('A,', '')

        # search for body
        # exit loop if body is empty
        body_start = rule.find(re_dcol)
        body = rule[body_start + len(re_dcol):]

        # get rule probability
        rule_prob = float(re.findall(re_float, body)[0])

        # construct list with predicates
        predicates = body.split(',')

        # iterate over predicates
        for predicate in predicates:

            # extracte predicate name
            predicate_end = predicate.find('(')
            predicate_name = predicate[:predicate_end].strip()

            # exit loop if body is empty
            if 'true' in predicate_name:
                continue

            # extract negation
            negation = False
            if '\\+' in predicate_name:
                negation = True
                predicate_name = predicate_name.replace('\\+', '')

            # extract value
            value = predicate[predicate_end:]
            value = value[value.find('(') + 1:value.find(')')]

            # add extractions as row to rule_df
            tmp = {'probability': [rule_prob],
                   'negation': [negation],
                   'predicate': [predicate_name],
                   'value': [value]}
            tmp = pd.DataFrame.from_dict(tmp)
            rule_df = pd.concat([rule_df, tmp])

        # exit loop if body of rule was empty
        if 'true' in predicate_name:
            continue

        # add rule_df to output list
        rule_df = rule_df.reset_index(drop=True)
        theory_df_list.append(rule_df)

    return theory_df_list
