import pandas as pd
import numpy as np
import random
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def find_best_fit(Folds_Dictionary, compound_pos_weight, compound_neg_weight, pos_class_fold_weight, neg_class_fold_weight):
    """
    Find which of the folds in the cross-validation should receive a compound,
    based on its pos/neg weights and the current weights in the folds
    :param Folds_Dictionary: dict object with the list of compounds in each fold, and their (balanced) positive and negative class weights
    :param compound_pos_weight: positive weight of current compound
    :param compound_neg_weight: negative weight of current compound
    :param pos_class_fold_weight: total positive weight of current fold
    :param neg_class_fold_weight: total negative weight of current fold
    :return: String name of the best fit fold (Fold1, Fold2 ... Fold5)
    """
    best_fit = "Fold1"
    min_overflow = 999
    for fold in Folds_Dictionary:
        fold_pos_weight = Folds_Dictionary[fold]["pos_weight"]
        fold_neg_weight = Folds_Dictionary[fold]["neg_weight"]
        pos_weight_overflow = (fold_pos_weight + compound_pos_weight) - pos_class_fold_weight
        neg_weight_overflow = (fold_neg_weight + compound_neg_weight) - neg_class_fold_weight
        if pos_weight_overflow < 0 and neg_weight_overflow < 0:
            return fold # compound completely fits inside the current fold, set it there
        elif pos_weight_overflow > 0 and neg_weight_overflow > 0:
            total_compound_overflow = pos_weight_overflow + neg_weight_overflow
            if total_compound_overflow < min_overflow:
                best_fit = fold
                min_overflow = total_compound_overflow
        elif pos_weight_overflow > 0:
            if pos_weight_overflow < min_overflow:
                best_fit = fold
                min_overflow = pos_weight_overflow
        else:
            if neg_weight_overflow < min_overflow:
                best_fit = fold
                min_overflow = neg_weight_overflow
    return best_fit


def create_Folds_Dictionary(df, random_seed, element_id_attribute):
    """
    Divide compounds from the dataset into folds, assuring the M and F entries for a compound are never separated.
    This avoids data leakage issues where a compound is used in both training and testing.
    :param df: pandas dataframe of the dataset
    :param random_seed: integer number used to randomly separate the data (experiments were with seeds 101-110)
    :param element_id_attribute: name of the feature used to name the compound. Default 'Compound_ID' (can be changed on main())
    :return: dict object with the list of compounds in each fold, and their (balanced) positive and negative class weights
    """
    compounds_list = list(df[element_id_attribute].unique())
    random.seed(random_seed)
    random.shuffle(compounds_list) #shuffle list of compounds so that dataset order has no bearing on result
    n_instances = df.shape[0]
    positive_class_count = df['class'].value_counts()[1]
    pos_class_fold_weight = round(positive_class_count/5, 2)
    neg_class_fold_weight = round(n_instances/5 - positive_class_count/5, 2)
    Folds_Dictionary = {"Fold1": {"pos_weight": 0, "neg_weight": 0, "compound_list": []},
            "Fold2": {"pos_weight": 0, "neg_weight": 0, "compound_list": []},
            "Fold3": {"pos_weight": 0, "neg_weight": 0, "compound_list": []},
            "Fold4": {"pos_weight": 0, "neg_weight": 0, "compound_list": []},
            "Fold5": {"pos_weight": 0, "neg_weight": 0, "compound_list": []}}

    for compound in compounds_list:
        compound_df = df[df[element_id_attribute] == compound]
        compound_neg_weight = compound_df[compound_df['class'] == 0].shape[0]
        compound_pos_weight = compound_df[compound_df['class'] == 1].shape[0]
        TargetFold = find_best_fit(Folds_Dictionary, compound_pos_weight, compound_neg_weight, pos_class_fold_weight, neg_class_fold_weight)
        Folds_Dictionary[TargetFold]["pos_weight"] = Folds_Dictionary[TargetFold]["pos_weight"] + compound_pos_weight
        Folds_Dictionary[TargetFold]["neg_weight"] = Folds_Dictionary[TargetFold]["neg_weight"] + compound_neg_weight
        Folds_Dictionary[TargetFold]["compound_list"].append(compound)
    return Folds_Dictionary


def train_RF(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model with 500 trees and default hyperparameters. Evaluate it on the parameter test set
    :param X_train: features in the training dataset
    :param X_test: features in the test dataset
    :param y_train: labels in the training dataset
    :param y_test: labels in the test dataset
    :return: evaluation metrics for current train/test datasets
    """
    if np.all(y_test == 0) or np.all(y_test == 1):
        print('Constant test set error.')
        return 0, 0, 0, 0, 0, 0
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', random_state=0)
    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_prob = rf.predict_proba(X_test)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob[:, 1])
    auc_precision_recall = auc(recall, precision)
    AUC = roc_auc_score(y_test, y_pred_prob[:, 1])  # sample_weight parameter if there was manual weights
    return round(TP,2), round(TN,2), round(FP,2), round(FN,2), round(AUC,3), round(auc_precision_recall, 3)


def print_metrics(TP_array, FP_array, TN_array, FN_array, AUC_array, PR_AUC_array, write_obj):
    """
    Print evaluation metrics for the current fold of a cross-validation, and adds them to the output write_obj file
    Each array has 5 values, for folds 1-5 of the cross-validation. The individual values are printed and then
    the mean (median for AUC and PR-AUC) values are printed in the summary line.
    :param TP_array: True-positives array
    :param FP_array: False-positives array
    :param TN_array: True-negatives array
    :param FN_array: False-negatives array
    :param AUC_array: Area under the ROC curve array
    :param PR_AUC_array: Area under the Precision-Recall curve array
    :param write_obj: tsv output file
    """
    print('Fold\tTP\tFP\tTN\tFN\tAUC\tPR_AUC')
    write_obj.write('Fold\tTP\tFP\tTN\tFN\tAUC\tPR_AUC\n')
    for i in range(0, 5):
        print(f'{i}\t{TP_array[i]}\t{FP_array[i]}\t{TN_array[i]}\t{FN_array[i]}\t{AUC_array[i]}\t{PR_AUC_array[i]}')
        write_obj.write(f'{i}\t{TP_array[i]}\t{FP_array[i]}\t{TN_array[i]}\t{FN_array[i]}\t{AUC_array[i]}\t{PR_AUC_array[i]}\n')
    print(f'Avg/Median\t{round(np.sum(TP_array),3)}\t{round(np.sum(FP_array),3)}\t{round(np.sum(TN_array),3)}\t{round(np.sum(FN_array),3)}\t{round(np.median(AUC_array),3)}\t{round(np.median(PR_AUC_array),3)}')
    write_obj.write(f'Avg/Median\t{round(np.sum(TP_array),3)}\t{round(np.sum(FP_array),3)}\t{round(np.sum(TN_array),3)}\t{round(np.sum(FN_array),3)}\t{round(np.median(AUC_array),3)}\t{round(np.median(PR_AUC_array),3)}\n')


def load_dataset(path, fixed_sex):
    """
    Preprocess the pandas dataset.
    Requires loading a dataset with an index, Compound_name and no other "header" features
    only predictive features and the class. Remove features such as Targets list prior to running.
    :param path: filepath for the tsv file of the pandas dataframe
    :param fixed_sex: Whether to filter the instances to include only 'M' or 'F' (male or female) instances, or keep 'all'
    :return: the fully preprocessed dataset
    """
    df = pd.read_csv(path, na_values='?', sep='\t', encoding='unicode_escape', index_col=0)
    #be sure to remove possible header features not used in internal code

    sex_values = {'M': 0, 'F': 1} #map string values into numeric values for sex variable
    df['sex'] = df['sex'].map(sex_values)
    if fixed_sex == 'M':
        print('Male-only dataset filter applied.')
        df = df.query('sex == 0')
        df = df.drop('sex', 1)
    elif fixed_sex == 'F':
        print('Female-only dataset filter applied.')
        df = df.query('sex == 1')
        df = df.drop('sex', 1)
    df = removeLowFrequencyFeatures(df, 3)
    return df


def removeInvalidInstances(df):
    """
    Part of preprocessing a dataframe for running the experiments.
    Drops instances that have all '0' values in the features, as they represent compounds without any valid data
    for classification.
    :param df: pandas dataframe of the dataset
    :return: the fully preprocessed dataset is returned to load_dataset
    """
    df = df.reset_index()
    df = df.drop('Index', axis=1)
    features_only_df = df.copy(deep=True)  # create a copy of the df including only valid binary features
    features_only_df = features_only_df.drop('class', axis=1)
    try:
        features_only_df = features_only_df.drop('sex', axis=1)
    except:
        print('Warning: sex feature not found when trying to remove it for removeLowFrequencyFeatures. Expected for M+F datasets.')
    try:
        features_only_df = features_only_df.drop('Compound_name', axis=1)
    except:
        print('Warning: Compound_name feature not found when trying to remove it for removeLowFrequencyFeatures.')
    drop_instances = []
    for key in range(len(features_only_df)):
        if features_only_df.iloc[key].sum() == 0:  # remove instances with only 0 values
            drop_instances.append(key)  # used reset_index to make sure it starts from 0, so no +1 needed here
    df = df.drop(drop_instances, axis=0)
    return df


def removeLowFrequencyFeatures(df, threshold):
    """
    Part of preprocessing a dataframe for running the experiments.
    Removes binary features that have fewer '1' or '0' values than the minimum threshold of variance
    :param df: pandas dataframe of the dataset
    :param threshold: integer threshold of variance
    :return: calls the next data preprocessing step for removing invalid instances from the dataframe
    """
    if threshold < 0:
        return df
    number_instances = df.shape[0]
    for feature in df:
        if feature != 'Compound_name':
            try:
                if df[feature].isin([0, 1]).all():
                    zero_counts = (df[feature] == 0).sum()
                    one_counts = (df[feature] == 1).sum()
                    if number_instances - zero_counts < threshold:
                        df = df.drop(feature, 1)
                    elif number_instances - one_counts < threshold:
                        df = df.drop(feature, 1)
                else:
                    print(f'Warning: skipped {feature} for having values different from 0 and 1')
            except:
                print(f'Error on {feature} when trying to apply minimum threshold filter')
    return removeInvalidInstances(df)


def cross_validation(df, random_seed, Compound_ID_feature, write_obj):
    """
    Runs a 5-fold cross-validation using the create_Folds_Dictionary() to divide the data
    :param df: pandas dataframe of the dataset
    :param random_seed: Integer random seed
    :param write_obj: tsv output file
    """
    write_obj.write(f'Instances: {df.shape[0]} - Features: {df.shape[1]}\n')
    TP_array = []
    FP_array = []
    TN_array = []
    FN_array = []
    AUC_array = []
    PR_AUC_array = []
    Folds_Dictionary = create_Folds_Dictionary(df, random_seed, Compound_ID_feature)
    for fold in Folds_Dictionary:
        training_set = df.loc[~df[Compound_ID_feature].isin(Folds_Dictionary[fold]['compound_list'])]
        test_set = df.loc[df[Compound_ID_feature].isin(Folds_Dictionary[fold]['compound_list'])]
        training_set = training_set.drop([Compound_ID_feature], 1) # drop compound name feature
        test_set = test_set.drop([Compound_ID_feature], 1) # drop compound name feature
        X_train, X_test = training_set.iloc[:, :-1], test_set.iloc[:, :-1]
        y_train, y_test = training_set.iloc[:, -1], test_set.iloc[:, -1]
        TP, TN, FP, FN, AUC, PR_AUC = train_RF(X_train, X_test, y_train, y_test)
        TN_array.append(TN)
        FP_array.append(FP)
        FN_array.append(FN)
        TP_array.append(TP)
        AUC_array.append(AUC)
        PR_AUC_array.append(PR_AUC)
    print_metrics(TP_array, FP_array, TN_array, FN_array, AUC_array, PR_AUC_array, write_obj)


def feature_importance_output(df, random_seed, write_obj):
    """
    Calculates the feature importances of a RF classifier for the input dataset and prints them in the write_obj file
    :param df: pandas dataframe of the dataset
    :param random_seed: Integer random seed
    :param write_obj: tsv output file
    """
    df = df.drop('Compound_name', 1)  # no train/test division
    X_train = df.iloc[:, :-1]
    X_test = df.iloc[:, -1]
    rf = RandomForestClassifier(n_estimators=500, random_state=random_seed, class_weight='balanced_subsample')
    rf = rf.fit(X_train, X_test)
    fi_array = rf.feature_importances_
    write_obj.write(f'Index\tFeature Name\tScore\n')
    for i, v in enumerate(fi_array):
        write_obj.write(f'{i}\t{df.columns[i]}\t{round(v,5)}\n')


def false_positive_analysis(df, random_seed, write_obj):
    """
    Prints out the positive-class likelihood of all instances in a dataset based on the default RF classifier
    for each fold of a cross-validation using the create_Folds_Dictionary() method
    :param df: pandas dataframe of the dataset
    :param random_seed: Integer random seed
    :param write_obj: tsv output file
    """
    Compound_ID_feature = 'Compound_name'  # required for proprietary train/test division in Weighted_StratifiedCV
    if 'sex' in df.columns:  # different output for M+F datasets, including sex information for each instance
        write_obj.write(f'Index\tFold\tCompound_name\tsex\tPosPredProb\tClass\n')
    else:
        write_obj.write(f'Index\tFold\tCompound_name\tPosPredProb\tClass\n')
    Folds_Dictionary = create_Folds_Dictionary(df, random_seed, Compound_ID_feature)
    for fold in Folds_Dictionary:
        training_set = df.loc[~df[Compound_ID_feature].isin(Folds_Dictionary[fold]['compound_list'])]
        test_set = df.loc[df[Compound_ID_feature].isin(Folds_Dictionary[fold]['compound_list'])]
        test_compound_names = test_set[Compound_ID_feature].tolist()
        training_set = training_set.drop([Compound_ID_feature], 1)
        test_set = test_set.drop([Compound_ID_feature], 1)
        X_train, X_test = training_set.iloc[:, :-1], test_set.iloc[:, :-1]
        y_train, y_test = training_set.iloc[:, -1], test_set.iloc[:, -1]
        rf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced_subsample')
        rf = rf.fit(X_train, y_train)
        y_pred_prob = rf.predict_proba(X_test)
        list_pred = y_pred_prob.tolist()
        list_y = y_test.tolist()
        list_index = X_test.index.tolist()
        if 'sex' in df.columns:
            test_sex = test_set['sex'].tolist()
        for i in range(0, len(list_pred)):  # Print prediction probabilities, with all relevant info for the instance
            correct_label = list_y[i]
            instance_index = list_index[i]
            instance_name = test_compound_names[i]
            pred_prob = list(list_pred[i])[1]
            if 'sex' in df.columns:
                sex = test_sex[i]
                write_obj.write(f'{instance_index}\t{fold}\t{instance_name}\t{sex}\t{pred_prob}\t{correct_label}\n')
            else:
                write_obj.write(f'{instance_index}\t{fold}\t{instance_name}\t{pred_prob}\t{correct_label}\n')


def main(input_file, output_file, fixed_sex, random_seed, run_cv, run_fi, run_fp):
    """
    Main run of the experimental setup. Change to include/exclude different experiments in the output write_obj:
    cross_validation: trains a 5-fold cross validation of a RF classifier and returns predictive accuracy results
    feature_importance_output: trains a RF model with the entire dataset and returns each feature's importance
    false_positive_analysis: prints the individual prediction probabilities of all compounds, for FP analysis

    :param input_file: Path to the dataset file
    :param output_file: Path to output file
    :param fixed_sex: 'all', 'M' or 'F'
    :param random_seed: Integer random seed
    :param run_cv: Boolean, run cross-validation
    :param run_fi: Boolean, run feature importance
    :param run_fp: Boolean, run false positive analysis
    """

    Compound_ID_feature = 'Compound_name'
    df = load_dataset(input_file, fixed_sex)

    with open(output_file, 'w', encoding="utf8") as write_obj:

        if run_cv:
            cross_validation(df, random_seed, Compound_ID_feature, write_obj)
            write_obj.write('\n')

        if run_fi:
            feature_importance_output(df, random_seed, write_obj)
            write_obj.write('\n')

        if run_fp:
            false_positive_analysis(df, random_seed, write_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experimental pipeline")

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input dataset file"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output file"
    )

    parser.add_argument(
        "-s", "--sex",
        choices=["all", "M", "F"],
        default="all",
        help="Dataset filter by sex"
    )

    parser.add_argument(
        "-r", "--random-seed",
        type=int,
        default=101,
        help="Random seed (default: 101)"
    )

    parser.add_argument(
        "-c", "--cross-validation",
        action="store_true",
        help="Run cross-validation"
    )

    parser.add_argument(
        "-f", "--feature-importance",
        action="store_true",
        help="Run feature importance analysis"
    )

    parser.add_argument(
        "-a", "--false-positive",
        action="store_true",
        help="Run false positive analysis"
    )

    args = parser.parse_args()

    # If no analysis flags are provided, run everything
    if not (args.cross_validation or args.feature_importance or args.false_positive):
        args.cross_validation = True
        args.feature_importance = True
        args.false_positive = True

    main(
        input_file=args.input,
        output_file=args.output,
        fixed_sex=args.sex,
        random_seed=args.random_seed,
        run_cv=args.cross_validation,
        run_fi=args.feature_importance,
        run_fp=args.false_positive
    )