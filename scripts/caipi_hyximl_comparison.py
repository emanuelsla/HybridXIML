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

from modules.distances.gower_distance import compute_distance_matrix_between_dfs
from modules.probfoil.utils import prepare_for_problog
from modules.probfoil.predict_probfoil import predict_probfoil
from modules.counterfactual_explainers.apply_dice import apply_dice


def convert_report_to_df(report):
    report_dict = {}
    for outer_key, outer_val in report.items():
        if isinstance(outer_val, dict):
            for inner_key, inner_val in outer_val.items():
                report_dict[str(outer_key.replace(' ', '_')) + '_' + str(inner_key)] = inner_val
        else:
            report_dict[outer_key] = outer_val
    df_result = pd.DataFrame(report_dict, index=[0])
    return df_result


def evaluate_counterfactual(ce, orig):
    changed = []
    for col in ce.columns:
        if int(ce[col].iloc[0]) != int(orig[col].iloc[0]):
            changed.append(col)
    changed = [True if col.startswith('spurious') else False for col in changed]
    return not any(changed)


if __name__ == '__main__':

    import argparse
    import os
    import pandas as pd
    import numpy as np
    from joblib import load
    from sklearn.metrics import classification_report
    from raiutils.exceptions import UserConfigValidationException

    # initialize flags
    parser = argparse.ArgumentParser('evaluate caipi and probfoil')
    parser.add_argument('-out', '--output_path', default='results/test.csv',
                        help='path of output data frame')

    # retrieve flags
    args_dict = vars(parser.parse_args())
    output_path = args_dict['output_path']

    # check if output directory is empty
    if os.path.isfile(output_path):
        raise(FileExistsError('output file already exists.'))

    # directories with data and models
    # (retrieved from main and generate theories)
    logging_dirs = ['models/test/']

    # list of classification models
    models = ['RandomForestClassifier']

    # random seed for each experimental iteration
    seeds = [42, 420, 4200, 42000, 420000]

    # gower distance values to evaluate
    gower_distances = list(np.arange(0.0, 5.0, 0.25))

    # initalize output data frame
    df_output = pd.DataFrame()

    # iterate over logging directories
    for logging_dir in logging_dirs:

        # obtain data, model, and explainer string from directory name
        log = logging_dir.split('/')[1]
        data = log.split(('_'))[0]
        model_str = log.split(('_'))[1]
        explainer = log.split(('_'))[2]

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

        char_cols_wo_y_name = nominal_column_names + ordinal_column_names
        char_cols_wo_y_name.remove(y_name)

        # get file list
        files = os.listdir(logging_dir)

        # sort files according to their content
        theory_files = [f for f in files if 'theorydict' in f]
        model_files = [f for f in files if any([True if m in f else False for m in models])]
        dftest_files = [f for f in files if 'dftest' in f]
        dfunlabeled_files = [f for f in files if 'dfunlabeled' in f]
        dfrwr_files = [f for f in files if 'dfrwr' in f]

        # sort each file group according to experimental and caipi iterations
        theory_files = sorted(theory_files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
        model_files = sorted(model_files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
        dftest_files = sorted(dftest_files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
        dfunlabeled_files = sorted(dfunlabeled_files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
        dfrwr_files = sorted(dfrwr_files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))

        # get variable transformers
        label_transformers_f = [f for f in files if 'labeltransformers' in f][0]
        metric_transformers_f = [f for f in files if 'metrictransformers' in f][0]
        label_transformers = load(logging_dir+label_transformers_f)
        metric_transformers = load(logging_dir+metric_transformers_f)

        # iterate over each file group by enumeration
        for i, _ in enumerate(theory_files):

            # load data, theory, and model
            df_rwr = pd.read_csv(logging_dir+dfrwr_files[i])
            df_unlabeled = pd.read_csv(logging_dir + dfunlabeled_files[i])
            df_test = pd.read_csv(logging_dir + dftest_files[i])
            theory_dict = load(logging_dir + theory_files[i])
            model = load(logging_dir + model_files[i])

            # get iterations and seed from file name
            experimental_iter = int(dfrwr_files[i].split('_')[0])
            caipi_iter = int(dfrwr_files[i].split('_')[1])
            seed = seeds[experimental_iter-1]

            # clean data frames
            cols = list(df_rwr.columns)
            drop_cols = [col for col in cols if 'Unnamed' in col]
            df_rwr = df_rwr.drop(drop_cols, axis=1)
            df_unlabeled = df_unlabeled.drop(drop_cols, axis=1)
            df_test = df_test.drop(drop_cols, axis=1)

            # go to next iteration if no rwr instance exists
            if len(df_rwr) == 0:
                print('no rwr instance, go to next iteration')
                print()
                continue

            # predict test data with ml model
            y_pred_ml = model.predict(df_test.drop(y_name, axis=1))
            df_test['y_pred_ml'] = y_pred_ml

            # initialize list for explanation evaluation
            corr_expls = [False] * len(df_test)

            # iterate over test instances
            for n in range(len(df_test)):

                # execute and evaluate dice
                try:
                    expl = apply_dice(model,
                                      df_test.iloc[[n]].drop('y_pred_ml', axis=1), df_unlabeled,
                                      list(df_test.iloc[[n]].drop([y_name, 'y_pred_ml'], axis=1).columns),
                                      y_name, number_cfs=1, seed=seed)
                    corr_expl = evaluate_counterfactual(expl, df_test.iloc[[n]].drop(['y_pred_ml'], axis=1))
                except (TimeoutError, UserConfigValidationException):
                    corr_expl = False

                # add result to list
                corr_expls[n] = corr_expl

            # add result list to data frame
            df_test['corr_expl_ml'] = corr_expls

            # add probfoil explanation results
            # they are always true, as probfoil is trained without spurious correlation
            df_test['corr_expl_probfoil'] = True

            # if there exists a theory
            # (at least one label and one rule)
            if len(theory_dict.keys()) > 0 and len(list(theory_dict.values())[0]) > 0:

                # prepare and predict with probfoil theory
                df_test_probfoil = prepare_for_problog(df_test, label_transformers,
                                                       metric_transformers, ordinal_encoding)
                y_pred_probfoil = predict_probfoil(theory_dict, df_test_probfoil)

                # re-encode probfoil predictions to numerics
                y_pred_probfoil = [
                    label_transformers[y_name].transform(np.reshape(y, (-1,)))[0] if y else 0 for y in y_pred_probfoil]

                # add result to data frame
                df_test['y_pred_probfoil'] = y_pred_probfoil

            # if not, probfoil prediction will be zero class
            else:
                df_test['y_pred_probfoil'] = 0

            # compute gower distance matrix between rwr and test data
            gower_distance_matrix = compute_distance_matrix_between_dfs(df_rwr, df_test,
                                                                        metric_column_names, char_cols_wo_y_name)

            # get minimum distance for each test instance to rwr data
            dist = np.min(gower_distance_matrix, axis=0)
            df_test['min_dist'] = dist

            # iterate over gower distances
            for gower_dist in gower_distances:

                print('logging directory:', str(logging_dir))
                print('experimental iteration:', str(experimental_iter))
                print('caipi iteration:', str(caipi_iter))
                print('gower distance:', str(gower_dist))
                print()

                # select subset below or equal to gower distance threshold
                df_test_similar = df_test[df_test['min_dist'] <= gower_dist]

                # generate predictive output of ml model
                report_ml = classification_report(df_test_similar[y_name], df_test_similar['y_pred_ml'],
                                                  zero_division=0, output_dict=True)
                df_result_ml = convert_report_to_df(report_ml)

                # select correct predictions of ml model
                df_eval_expl = df_test_similar.copy()
                df_eval_expl = df_eval_expl[df_eval_expl[y_name] == df_eval_expl['y_pred_ml']]

                # get ratio of correct explanation wrt. to correct prediction and label
                df_eval_expl_group = df_eval_expl[[y_name, 'corr_expl_ml']].groupby(y_name).sum() / len(df_eval_expl)

                # add explanatory output of ml model to result data
                report_ml = {}
                for n in range(len(df_eval_expl_group)):
                    report_ml[str(df_eval_expl_group.index[n]) + '_corr_expl_ml'] \
                        = df_eval_expl_group['corr_expl_ml'].iloc[n]
                df_result_ml = pd.concat([df_result_ml, pd.DataFrame(report_ml, index=[0])], axis=1)

                # indicate ml method as variable
                df_result_ml['method'] = 'ml'

                # generate predictive result of probfoil
                report_probfoil = classification_report(df_test_similar[y_name], df_test_similar['y_pred_probfoil'],
                                                        zero_division=0, output_dict=True)
                df_result_probfoil = convert_report_to_df(report_probfoil)

                # select correct predictions of probfoil
                df_eval_expl = df_test_similar.copy()
                df_eval_expl = df_eval_expl[df_eval_expl[y_name] == df_eval_expl['y_pred_probfoil']]

                # get ratio of correct explanation wrt. to correct prediction and label
                df_eval_expl_group = df_eval_expl[[y_name, 'corr_expl_probfoil']].groupby(y_name).sum() \
                    / len(df_eval_expl)

                # add explanatory output of probfoil model to result data
                report_probfoil = {}
                for n in range(len(df_eval_expl_group)):
                    report_probfoil[str(df_eval_expl_group.index[n]) + '_corr_expl_probfoil'] \
                        = df_eval_expl_group['corr_expl_probfoil'].iloc[n]
                df_result_probfoil = pd.concat([df_result_probfoil, pd.DataFrame(report_probfoil, index=[0])], axis=1)

                # indicate probfoil method as variable
                df_result_probfoil['method'] = 'probfoil'

                # concat ml and probfoil result data sets
                df_result = pd.concat([df_result_ml, df_result_probfoil])

                # add seed, iteration, and model and explainer strings
                df_result['seed'] = seed
                df_result['iteration'] = caipi_iter
                df_result['classifier'] = model_str
                df_result['explainer'] = explainer

                # append result to output data frame and reset index
                df_output = pd.concat([df_output, df_result]).reset_index(drop=True)

            print('current output df:')
            print()
            print(df_output)
            print()

            df_output.to_csv(output_path)
            print('updated file')
            print()
