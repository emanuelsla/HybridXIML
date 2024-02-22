from modules.preprocessors.preprocess_diabetes_data import preprocess_diabetes_data, \
    y_name, nominal_column_names, ordinal_column_names, metric_column_names, ordinal_encoding
diabetes_y_name = y_name
diabetes_nominal_column_names = nominal_column_names
diabetes_ordinal_column_names = ordinal_column_names
diabetes_metric_column_names = metric_column_names
diabetes_ordinal_encoding = ordinal_encoding
from modules.preprocessors.preprocess_diagnostic_data import preprocess_diagnostic_data, \
    y_name, nominal_column_names, ordinal_column_names, metric_column_names, ordinal_encoding
diagnostic_y_name = y_name
diagnostic_nominal_column_names = nominal_column_names
diagnostic_ordinal_column_names = ordinal_column_names
diagnostic_metric_column_names = metric_column_names
diagnostic_ordinal_encoding = ordinal_encoding

from modules.probfoil.utils import prepare_for_problog
from modules.probfoil.train_probfoil import train_probfoil


if __name__ == '__main__':

    import os
    import argparse
    import pandas as pd
    import numpy as np
    from joblib import dump

    parser = argparse.ArgumentParser('generate theories from rwr data frames')

    parser.add_argument('-dir', '--directory', default='models/test/',
                        help='directory')
    parser.add_argument('-p', '--problog_path', default='scripts/autonomous_code.pl',
                        help='path to problog file, if None execution without ProbFOIL')
    args_dict = vars(parser.parse_args())

    problog_path = args_dict['problog_path']

    directory = str(args_dict['directory'])

    # check if directory exists
    if not os.path.isdir(directory):
        raise (FileNotFoundError('directory does not exist'))

    # select and sort rwr data frames
    files = os.listdir(directory)
    rwr_names = [f for f in files if 'dfrwr' in f]
    rwr_names = sorted(rwr_names, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))

    # initialize variables for loop
    df_rwr_old = pd.DataFrame()
    theory_dict = {'some_label': []}
    current_iter = 1

    # seeds for experimental iterations
    seeds = [42, 420, 4200, 42000, 420000]

    # retrieve data name from directory name
    data = directory.split('/')[-2]
    data = data.split('_')[0]

    if data == 'diabetes':
        path = 'data/diabetes.csv'
        preprocessor = preprocess_diabetes_data
        y_name = diabetes_y_name
        nominal_column_names = diabetes_nominal_column_names
        ordinal_column_names = diabetes_ordinal_column_names
        metric_column_names = diabetes_metric_column_names
        ordinal_encoding = diabetes_ordinal_encoding

    elif data == 'diagnostic':
        path = 'data/diagnostic.csv'
        preprocessor = preprocess_diagnostic_data
        y_name = diagnostic_y_name
        nominal_column_names = diagnostic_nominal_column_names
        ordinal_column_names = diagnostic_ordinal_column_names
        metric_column_names = diagnostic_metric_column_names
        ordinal_encoding = diagnostic_ordinal_encoding

    # iterate over rwr data frames
    for name in rwr_names:

        print('file:')
        print(name)
        print()

        # reset loop initialization if new seed starts
        seed_index = int(name.split('_')[0]) - 1
        if seed_index != current_iter:
            theory_dict = {'some_label': []}
            df_rwr_old = pd.DataFrame()
            current_iter = seed_index

        # read rwr data frame
        df_rwr = pd.read_csv(directory+name)

        # delete unneeded cols, especially spurious correlation
        cols = list(df_rwr.columns)
        drop_cols = [col for col in cols if 'spurious' in col]
        drop_cols += [col for col in cols if 'Unnamed' in col]
        df_rwr = df_rwr.drop(drop_cols, axis=1)

        # check if data frame has changed from last file
        if df_rwr.equals(df_rwr_old):
            print('same df, continue')
            print()

        # if data frame has changed, re-generate theory
        else:
            print(df_rwr)
            print()

            # execute pre-processor with specific seed to get transformers
            df_unlabeled, df_test, label_transformers, metric_transformers = preprocessor(path, test_ratio=0.3,
                                                                                          seed=seeds[seed_index])

            # execute pre-processing for problog
            df_train_probfoil_prep = prepare_for_problog(df_rwr, label_transformers,
                                                         metric_transformers, ordinal_encoding)

            # train with probfoil algorithm
            theory_dict_new = train_probfoil(df_train_probfoil_prep, y_name,
                                             list(np.unique(df_train_probfoil_prep[y_name])),
                                             seeds[seed_index], problog_path)
            print(theory_dict_new)
            print()

            # overwrite theory with new one
            theory_dict = theory_dict_new

        # overwrite df with new one
        df_rwr_old = df_rwr

        # save theory
        theory_name = '_'.join(name.split('_')[:-1]) + '_theorydict.joblib'
        dump(theory_dict, directory + theory_name)

        print('save theory:')
        print(theory_name)
        print()
