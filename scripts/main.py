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

from modules.utils import generate_random_df
from modules.utils import insert_spurious_correlations
from modules.counterfactual_explainers.apply_dice import apply_dice
from modules.probfoil.utils import prepare_for_problog
from modules.probfoil.train_probfoil import train_probfoil
from modules.probfoil.predict_probfoil import predict_probfoil
from modules.distances.gower_distance import compute_distance_matrix_between_dfs


def evaluate_counterfactual(ce, orig):
    changed = []
    for col in ce.columns:
        if int(ce[col].iloc[0]) != int(orig[col].iloc[0]):
            changed.append(col)
    changed = [True if col.startswith('spurious') else False for col in changed]
    return not any(changed)


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


if __name__ == '__main__':

    import os
    import argparse
    import numpy as np
    import pandas as pd
    from joblib import dump

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from raiutils.exceptions import UserConfigValidationException

    parser = argparse.ArgumentParser('execute hyXIML')

    parser.add_argument('-d', '--data', default='diabetes',
                        help='data set')
    parser.add_argument('-o', '--output', default='results/test.csv',
                        help='output file')
    parser.add_argument('-l', '--logging', default='models/test/',
                        help='logging directory')
    parser.add_argument('-i', '--iterations', default=1,
                        help='experimental iterations')
    parser.add_argument('-hi', '--hyximl_iterations', default=20,
                        help='experimental iterations')
    parser.add_argument('-c', '--counterexamples', default=5,
                        help='number of counterexamples')
    parser.add_argument('-g', '--gower', default=0.25,
                        help='gower distance threshold')
    parser.add_argument('-p', '--problog_path', default='scripts/autonomous_code.pl',
                        help='path to problog file, if None execution without ProbFOIL')

    args_dict = vars(parser.parse_args())

    exp_iters = int(args_dict['iterations'])
    seeds = [42] * exp_iters
    for i in range(exp_iters):
        seeds[i] = seeds[i] * 10**i

    hyximl_iters = int(args_dict['hyximl_iterations'])

    c = int(args_dict['counterexamples'])
    gower_distance_threshold = float(args_dict['gower'])

    classifier_string = 'randomforest'
    explainer = 'counterfactuals'

    output_path = str(args_dict['output'])
    if os.path.isfile(output_path):
        raise(FileExistsError('output file already exists.'))

    logging_path = str(args_dict['logging'])
    if logging_path == 'None':
        logging_path = None
    if logging_path:
        if os.path.isdir(logging_path):
            raise(FileExistsError('logging directory already exists'))
        else:
            os.mkdir(logging_path)

    data_path = str(args_dict['data'])

    problog_path = str(args_dict['problog_path'])
    if problog_path == 'None':
        problog_path = None

    if args_dict['data'] == 'diabetes':
        path = 'data/diabetes.csv'
        preprocessor = preprocess_diabetes_data
        y_name = diabetes_y_name
        nominal_column_names = diabetes_nominal_column_names
        ordinal_column_names = diabetes_ordinal_column_names
        metric_column_names = diabetes_metric_column_names
        ordinal_encoding = diabetes_ordinal_encoding

    elif args_dict['data'] == 'diagnostic':
        path = 'data/diagnostic.csv'
        preprocessor = preprocess_diagnostic_data
        y_name = diagnostic_y_name
        nominal_column_names = diagnostic_nominal_column_names
        ordinal_column_names = diagnostic_ordinal_column_names
        metric_column_names = diagnostic_metric_column_names
        ordinal_encoding = diagnostic_ordinal_encoding

    else:
        raise(ValueError('Unknown data set. Inspect data folder for valid data sets.'))

    nr_spurious = 1
    spurious_cols = ['spurious_' + str(n + 1) for n in range(nr_spurious)]

    theory_dict = {'some_label': []}
    y_pred_probfoil = []
    ratio_probfoil_preds = 0

    print('classifier:', str(classifier_string))
    print('explainer:', str(explainer))
    print('experimental iterations:', str(exp_iters))
    print('hyximl iterations:', str(hyximl_iters))
    print('counterexamples:', str(c))
    print()

    df_output = pd.DataFrame()

    # iterate over experimental iterations
    for i, seed in enumerate(seeds):

        print('#### experimental iteration ' + str(i + 1) + ' ####')
        print()

        print('seed:', str(seed))
        print()

        np.random.seed(seed)

        # load data
        df_unlabeled, df_test, label_transformers, metric_transformers = preprocessor(path, test_ratio=0.3, seed=seed)
        dump(label_transformers, logging_path + 'labeltransformers.joblib')
        dump(metric_transformers, logging_path + 'metrictransformers.joblib')

        # for large data sets, use only 1,000 test instances
        if len(df_test) > 1000:
            df_test = df_test.sample(n=1000, random_state=seed)

        # categorical encoding for anchor
        cat_enc = {}
        for col, le in label_transformers.items():
            if col != y_name:
                cat_enc[list(df_unlabeled.drop(y_name, axis=1).columns).index(col)] = list(le.classes_)

        # generate random labeled data
        df_labeled = generate_random_df(df_unlabeled, y_name, size=100, seed=seed)

        # insert spurious correlations
        df_labeled = insert_spurious_correlations(df_labeled, y_name, nr_spurious=nr_spurious, seed=seed)
        df_unlabeled = insert_spurious_correlations(df_unlabeled, y_name, nr_spurious=nr_spurious, seed=seed)
        df_test = insert_spurious_correlations(df_test, y_name, nr_spurious=nr_spurious, seed=seed)

        # randomize spurious correlation on unlabeled data
        for col in list(df_unlabeled.columns):
            if col.startswith('spurious'):
                pop = list(np.unique(df_unlabeled[col]))
                df_unlabeled[col] = np.random.choice(pop, len(df_unlabeled))

        # randomize spurious correlation on test data
        for col in list(df_test.columns):
            if col.startswith('spurious'):
                pop = list(np.unique(df_test[col]))
                df_test[col] = np.random.choice(pop, len(df_test))

        # prepare data for probfoil
        df_train_probfoil = pd.DataFrame()
        df_test_probfoil = prepare_for_problog(df_test, label_transformers,
                                               metric_transformers, ordinal_encoding)
        gower_distance = [9999] * len(df_test)

        # prepare empty data set for rwr instances
        df_rwr = pd.DataFrame()

        # define classification model
        classifier = RandomForestClassifier(random_state=seed, class_weight='balanced')

        # start hyximl optimization
        for j in range(hyximl_iters):

            print('#### hyximl iteration ' + str(j + 1) + ' ####')
            print()

            # train classifier
            df_labeled = df_labeled.reset_index(drop=True)
            model = classifier
            model.fit(df_labeled.drop(y_name, axis=1), df_labeled[y_name])

            # predict unlabeled data
            y_pred = model.predict_proba(df_unlabeled.drop(y_name, axis=1))

            # select most-informative instance
            nr_classes = len(np.unique(df_unlabeled[y_name]))
            mii_dist = np.sum(np.absolute(np.subtract(y_pred, 1 / nr_classes)), axis=1)
            mii_index = np.argmin(mii_dist)
            mii = df_unlabeled.iloc[[mii_index]]
            mii_index = mii.index[0]

            # evaluate prediction
            y_pred = model.predict(mii.drop(y_name, axis=1))[0]

            if int(y_pred) == int(mii.loc[mii_index, y_name]):

                # generate and evaluate counterfactual explanation
                try:
                    expl = apply_dice(model, mii, df_unlabeled,
                                      list(mii.drop(y_name, axis=1).columns), y_name, number_cfs=1,
                                      seed=seed)
                    corr_expl = evaluate_counterfactual(expl, mii)
                except (TimeoutError, UserConfigValidationException):
                    corr_expl = False

                if corr_expl:
                    print('Right for the Right Reasons (RRR)')
                    print()
                    state = 'rrr'

                    df_labeled = pd.concat([df_labeled, mii])
                    df_unlabeled = df_unlabeled.drop(mii_index, axis=0)

                else:
                    print('Right for the Wrong Reasons (RWR)')
                    print()
                    state = 'rwr'

                    if problog_path:
                        # train probfoil without spurious columns

                        mii_probfoil = mii.drop(spurious_cols, axis=1)

                        df_train_probfoil = pd.concat([df_train_probfoil, mii_probfoil]).reset_index(drop=True)

                        df_train_probfoil_prep = prepare_for_problog(df_train_probfoil, label_transformers,
                                                                     metric_transformers, ordinal_encoding)

                        theory_dict = train_probfoil(df_train_probfoil_prep, y_name,
                                                     list(np.unique(df_train_probfoil_prep[y_name])),
                                                     seed, problog_path)

                        # predict probfoil
                        # if theory is complete and if each theory contains rules
                        theory_labels = set(theory_dict.keys())
                        rules_in_theory_bool = [True if len(t) > 0 else False for t in list(theory_dict.values())]
                        labels = set(df_test_probfoil[y_name])
                        if len(labels - theory_labels) == 0 and all(rules_in_theory_bool):
                            y_pred_probfoil = predict_probfoil(theory_dict, df_test_probfoil)

                            # calculate gower distance to determine the instances
                            # whose predictions shall be overruled by probfoil
                            char_cols_wo_y_name = nominal_column_names + ordinal_column_names
                            char_cols_wo_y_name.remove(y_name)
                            gower_distance_tmp = list(compute_distance_matrix_between_dfs(mii_probfoil,
                                                                                          df_test,
                                                                                          metric_column_names,
                                                                                          char_cols_wo_y_name)[0, :])
                            gower_distance = [gower_distance_tmp[n] if gower_distance_tmp[n] < gower_distance[n]
                                              else gower_distance[n] for n in range(len(gower_distance))]

                            probfoil_instance = [True if (g < gower_distance_threshold and y)
                                                 else False for y, g in zip(y_pred_probfoil, gower_distance)]
                            ratio_probfoil_preds = sum(probfoil_instance) / len(probfoil_instance)
                            print('ratio of predictions by probfoil:', ratio_probfoil_preds)
                            print()

                            # encode probfoil predictions
                            y_pred_probfoil = [
                                label_transformers[y_name].transform(np.reshape(y, (-1,)))[0] if g else -1
                                for y, g in zip(y_pred_probfoil, probfoil_instance)]

                    # generate counterexamples
                    df_ce = pd.DataFrame()
                    for n in range(c):
                        df_tmp = mii.copy()
                        for col in df_tmp:
                            if col.startswith('spurious'):
                                pop = list(np.unique(df_unlabeled[col]))
                                df_tmp[col] = np.random.choice(pop)
                        df_ce = pd.concat([df_ce, df_tmp])
                    df_ce = df_ce.reset_index(drop=True)

                    df_labeled = pd.concat([df_labeled, mii, df_ce])
                    df_unlabeled = df_unlabeled.drop(mii_index, axis=0)
                    df_rwr = pd.concat([df_rwr, mii])

            else:
                print('Wrong for the wrong reasons (W)')
                print()
                state = 'w'

                df_labeled = pd.concat([df_labeled, mii])
                df_unlabeled = df_unlabeled.drop(mii_index, axis=0)

            # save rwr data frame and model every 10-th iteration
            if logging_path and (j+1) % 10 == 0:
                df_rwr.to_csv(logging_path+str(i+1)+'_'+str(j+1)+'_dfrwr.csv')
                df_labeled.to_csv(logging_path + str(i + 1) + '_' + str(j + 1) + '_dflabeled.csv')
                df_unlabeled.to_csv(logging_path + str(i + 1) + '_' + str(j + 1) + '_dfunlabeled.csv')
                df_test.to_csv(logging_path + str(i + 1) + '_' + str(j + 1) + '_dftest.csv')
                dump(model, logging_path+str(i+1)+'_'+str(j+1)+'_'+str(type(model).__name__)+'.joblib')
                if problog_path:
                    dump(theory_dict, logging_path + str(i+1) + '_' + str(j+1) + '_theorydict.joblib')

            # evaluate predictive quality
            y_pred = classifier.predict(df_test.drop(y_name, axis=1))
            if len(y_pred_probfoil) > 0:
                y_pred = [y if yp == -1 else yp for y, yp in zip(y_pred, y_pred_probfoil)]
            report = classification_report(df_test[y_name], y_pred, zero_division=0)
            print(report)
            print()
            report = classification_report(df_test[y_name], y_pred, zero_division=0, output_dict=True)
            df_result = convert_report_to_df(report)

            # evaluate explanatory quality for correctly predicted instances
            df_eval_expl = df_test.copy()
            df_eval_expl['y_pred'] = y_pred
            df_eval_expl = df_eval_expl[df_eval_expl[y_name] == df_eval_expl['y_pred']]

            if len(y_pred_probfoil) > 0:
                df_eval_expl_probfoil = df_test.copy()
                df_eval_expl_probfoil['y_pred'] = y_pred
                df_eval_expl_probfoil['probfoil'] = probfoil_instance
                df_eval_expl_probfoil = df_eval_expl_probfoil[df_eval_expl_probfoil[y_name]
                                                              == df_eval_expl_probfoil['y_pred']]

            corr_expls = [False] * len(df_eval_expl)
            for n in range(len(df_eval_expl)):

                if len(y_pred_probfoil) > 0 and df_eval_expl_probfoil['probfoil'].iloc[n]:
                    corr_expl = True

                else:
                    try:
                        expl = apply_dice(model,
                                          df_eval_expl.iloc[[n]].drop('y_pred', axis=1), df_unlabeled,
                                          list(df_eval_expl.iloc[[n]].drop([y_name, 'y_pred'], axis=1).columns),
                                          y_name, number_cfs=1, seed=seed)

                        corr_expl = evaluate_counterfactual(expl, df_eval_expl.iloc[[n]].drop(['y_pred'], axis=1))
                    except (TimeoutError, UserConfigValidationException):
                        corr_expl = False

                corr_expls[n] = corr_expl

            df_eval_expl['corr_expl'] = corr_expls
            df_eval_expl_group = df_eval_expl[[y_name, 'corr_expl']].groupby(y_name).sum() / len(df_eval_expl)

            report = {}
            for n in range(len(df_eval_expl_group)):
                report[str(df_eval_expl_group.index[n]) + '_corr_expl'] = df_eval_expl_group['corr_expl'].iloc[n]

            included = [int(k.replace('_corr_expl', '')) for k in report.keys()]
            y_values = list(np.unique(df_test[y_name]))
            missing = list(set(y_values) - set(included))
            if len(missing) > 0:
                for val in missing:
                    report[str(val) + '_corr_expl'] = 0.0

            df_result = pd.concat([df_result, pd.DataFrame(report, index=[0])], axis=1)
            print('explanation evaluation:')
            print(pd.DataFrame(report, index=[0]))
            print()

            print('labeled size:', len(df_labeled))
            print('unlabeled size:', len(df_unlabeled))
            print()

            df_result['seed'] = seed
            df_result['iteration'] = j
            df_result['state'] = state
            df_result['labeled'] = len(df_labeled)
            df_result['unlabeled'] = len(df_unlabeled)
            df_result['classifier'] = classifier_string
            df_result['ratio_probfoil_preds'] = ratio_probfoil_preds
            df_result['explainer'] = explainer
            df_output = pd.concat([df_output, df_result], axis=0).reset_index(drop=True)

    df_output.to_csv(output_path)
