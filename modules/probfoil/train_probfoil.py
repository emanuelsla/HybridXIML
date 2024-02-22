from .utils import autonomous_code, execute_shell, postprocess_probfoil


def train_probfoil(df, y_name, y_name_values, seed, problog_path):
    command = 'probfoil'
    args_list = ['-v', '-s ' + str(seed)]

    if len(y_name_values) == 1:
        print('train_probfoil: No negative examples to learn ProbFOIL theory.')
        print()
        df_theory = {y_name_values[0]: []}
        return df_theory

    theory_dict = {}

    for val in y_name_values:
        code = autonomous_code(df, (y_name, val))
        with open(problog_path, 'w') as f:
            f.write(code)

        result = execute_shell(command, problog_path, args_list)

        df_theory = postprocess_probfoil(result)

        theory_dict[val] = df_theory

    return theory_dict

