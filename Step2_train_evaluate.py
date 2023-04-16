import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import ParameterGrid

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)

# warnings
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
#
# timing variable
timing_checkpoint = None

# Cost matrix:
C_TP, C_FN, C_FP, C_TN = -88, 0, 25.5, 0

# Expected Value per Positive Classification score
def EVPC_score(y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted, labels=[True, False])
    TP = cm[0][0]
    FP = cm[1][0]

    precision = TP / (TP + FP)

    return (C_TP * precision) + (C_FP * (1 - precision))


# cost score defined as "extracted value per customer"
def cost_score(y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted, labels=[True, False])
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    return (TP * C_TP + FN * C_FN + FP * C_FP + TN * C_TN) / (len(y_true))


# cost score defined as "extracted value / max extractable value"
def cost_extracted_score(y_true, y_predicted, kwars=None):
    cm = confusion_matrix(y_true, y_predicted, labels=[True, False])
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    max_cost = float(y_true.sum()) * C_TP

    return (TP * C_TP + FN * C_FN + FP * C_FP + TN * C_TN) / max_cost

def inv_cost_extracted_score(y_true, y_predicted, kwars=None):
    return 1.0 - cost_extracted_score(y_true, y_predicted, kwars)

# feature select, cross validate, and train model
def train_model(X_train, y_train, modelClass, paramgrid, feature_selection=True):
    results = []

    # expand parameter grid into list of distinct hyper-parameters
    hyperparameters = []
    for dict in paramgrid:
        params = ParameterGrid(dict)
        for p in params:
            hyperparameters.append(p)

    # add default parameters
    hyperparameters.insert(0, {})
    # test all combination of parameters
    for index, params in enumerate(hyperparameters):
        # only count non-default params
        if index > 0:
            print(f"\rCross-validating hyperparams: {index:03d} / {len(hyperparameters) - 1}", end="")
        count = 0
        avg_score = 0

        # forward feature selection
        if feature_selection:
            instance = modelClass(**params)
            # print("Forward selecting features...", end='')
            sfs = SequentialFeatureSelector(instance, n_features_to_select='auto', tol=0.01, cv=5, direction='forward',
                                            scoring=make_scorer(cost_extracted_score))
            sfs = sfs.fit(X_train, y_train)

            selected = sfs.get_support(indices=True)
            selected = [X_train.columns[x] for x in selected]

            # print("\rSelected feautures: ", end='')
            # print(selected)
            # print()

            X_train_reduced = sfs.transform(X_train)
        else:
            X_train_reduced = X_train
            selected = list(X_train.columns)

        # Stratief K-fold cross validation of parameters
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        for train_indices, validation_indices in kfold.split(X_train_reduced, y_train):
            # split dataset into folds
            X_train_fold = pd.DataFrame(X_train_reduced).iloc[train_indices]
            y_train_fold = y_train.iloc[train_indices]

            X_val_fold = pd.DataFrame(X_train_reduced).iloc[validation_indices]
            y_val_fold = y_train.iloc[validation_indices]

            # test model
            model = modelClass(**params)
            model.fit(X_train_fold, y_train_fold)

            y_val_predictions = model.predict(X_val_fold)

            avg_score += cost_extracted_score(y_val_fold, y_val_predictions)
            count += 1

        avg_score /= count

        # print unoptimised score
        if index == 0:
            print(f"Unoptimised validation extracted value: {avg_score} \n")

        results.append({'params': params, 'score': avg_score, 'features': selected})

    print()
    # return parameters with highest cost
    return max(results, key=lambda x: x['score'])


def evaluate_model(name, Model, hyperparameters, X_train, y_train, X_test, y_test, feature_selection=True):
    # base model
    print(f"==== {name} model ====")

    best_params = train_model(X_train, y_train, Model, hyperparameters, feature_selection)

    print()
    print(
        f"Validation extracted value: {best_params['score']} \n with strategy {best_params['params']} \n with features {best_params['features']} \n")
    model = Model(**best_params['params'])
    model.fit(X_train, y_train)

    y_test_predictions = model.predict(X_test)

    print(f"test F1: {f1_score(y_test, y_test_predictions)}")
    print(f"test AUC ROC score: {roc_auc_score(y_test, y_test_predictions)}")
    print(f"test precision: {precision_score(y_test, y_test_predictions)}")
    print(f"test recall: {recall_score(y_test, y_test_predictions)}")
    print(f"test accuracy: {accuracy_score(y_test, y_test_predictions)}")
    print()
    print(f"test average extracted value: {cost_score(y_test, y_test_predictions)}")
    print(f"test extracted value coefficient: {cost_extracted_score(y_test, y_test_predictions)}")
    print(f"test EVPC score: {EVPC_score(y_test, y_test_predictions)}")
    print()

    print(confusion_matrix(y_test, y_test_predictions, labels=[True, False]))
    print()
    print("========")
    print()
    global timing_checkpoint
    print("--- Evaluated in %s seconds ---\n" % int(time.time() - timing_checkpoint))
    timing_checkpoint = time.time()


def main(correlated_features, models, seed=None, best=False):
    dataset = pd.read_csv("data/existing-customers-CLEAN.csv", index_col=0)

    # Using Kendall Correlation to filter out non relevant feautures
    cor = dataset.corr(method="kendall")
    print("=== Feature correlations to class, treshhold 0.2 ===")
    # Correlation with output variable
    cor_target = cor["class"]  # Selecting highly correlated features
    relevant_features = cor_target[np.abs(cor_target) > 0.2]
    print(relevant_features)
    print()

    # split X and Y
    X = dataset.loc[:, dataset.columns != 'class']
    y = dataset.loc[:, dataset.columns == 'class']

    # Split train and test set, test set never used in training
    X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, shuffle=True, random_state=seed)
    X_train = X_train_full.copy()
    X_test = X_test_full.copy()

    # === remove non-relevant features ===
    if correlated_features:
        relevant_features = relevant_features.index.values.tolist()
        relevant_features.remove("class")
        relevant_features.remove("sex_Female")
        X_train.drop(columns=[col for col in X if col not in relevant_features], inplace=True)
        X_test.drop(columns=[col for col in X if col not in relevant_features], inplace=True)

    ### Evaluating different models below ###
    print("--- Starting Model Tests ---\n")
    global timing_checkpoint
    timing_checkpoint = time.time()
    start_time = timing_checkpoint

    # base model
    if "Base" in models:

        base_params = [
            {"strategy": ["constant"], "constant": [False, True]},
            {"strategy": ["uniform", "prior", "stratified"]},
        ]
        if best:
            base_params = [{'constant': [True], 'strategy': ['constant']}]

        evaluate_model("Base", DummyClassifier, base_params, X_train, y_train, X_test, y_test, feature_selection=not correlated_features)

    # KNN
    if "KNN" in models:
        knn_params = [
            {"weights": ["uniform", "distance"], "n_neighbors": [i for i in range(1, 21)]}
        ]
        if best:
            knn_params = [{'n_neighbors': [16], 'weights': ['distance']}]
        evaluate_model("KNN", KNeighborsClassifier, knn_params, X_train, y_train, X_test, y_test, feature_selection=not correlated_features)

    # Decision Tree model
    if "DecisionTree" in models:
        decision_params = [{"max_depth": [i for i in range(4, 21)]}]
        if best:
            decision_params = [{'max_depth': [9]}]
        evaluate_model("Decision Tree", DecisionTreeClassifier, decision_params, X_train, y_train, X_test, y_test, feature_selection=not correlated_features)


    # Random Forest
    if "RandomForest" in models:
        params = [
            {
                'class_weight': ['balanced'],
                'max_samples': [i * 0.1 for i in range(1, 11)],
                'max_features': [i for i in range(1, 15)],
                'n_estimators': [10, 50, 100, 500, 1000, 2000, 5000],
                'max_depth': [i for i in range(1, 15)] + [None],
                'n_jobs': [-1],
            }
        ]
        if best:
            params = [{'class_weight': ['balanced'], 'max_depth': [10], 'max_features': [10], 'max_samples': [0.1], 'n_estimators': [2000], 'n_jobs': [-1]}]
        evaluate_model("RF", RandomForestClassifier, params, X_train_full, y_train, X_test_full, y_test, feature_selection=False)

    # Gradient Boosting
    if "GradBoost" in models:
        gb_params = [
            {
                'class_weight': ['balanced'],
                'max_depth': [i for i in range(1, 15)] + [None],
                'max_iter': [10, 50, 100, 500, 1000],
                'categorical_features': [[col for col in X if col not in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]]
            }
        ]
        if best:
            gb_params = [{'categorical_features': [[col for col in X if col not in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]],
                           'class_weight': ['balanced'],
                          'max_depth': [4], 'max_iter': [500]
                          }]
        evaluate_model("Hist Gradient Boosting Classifier", HistGradientBoostingClassifier, gb_params, X_train_full,
                       y_train, X_test_full, y_test, feature_selection=False)

    # Linear Discriminant Analysis
    if "LinearDA" in models:
        ld_params = [
            {'solver': ['lsqr', 'eigen'], 'shrinkage': [i*0.05 for i in range(21)] + ['auto']},
            {'solver': ['svd']},
        ]
        if best:
            ld_params = [{'shrinkage': [0.45], 'solver': ['lsqr']}]
        evaluate_model("Linear Discriminant Analysis", LinearDiscriminantAnalysis, ld_params, X_train, y_train, X_test, y_test, feature_selection=not correlated_features)

    # Logistic Regression
    if "LogisticReg" in models:
        c_range = [100, 20, 10.0] + [i*0.1 for i in range(1,11)]
        lr_params = [
            {'solver': ['newton-cg', 'saga', 'sag', 'liblinear']},
            {'class_weight': ['balanced'], 'solver': ['newton-cg', 'saga', 'lbfgs', 'sag', 'liblinear']},
            {'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': [i*0.1 for i in range(11)], 'class_weight': ['balanced', None]},
            {'solver': ['newton-cg', 'saga', 'lbfgs', 'sag'], 'max_iter': [1000], 'penalty':['none', 'l2'], 'C': c_range, 'class_weight': ['balanced', None]},
            {'solver': ['liblinear'], 'max_iter': [1000], 'penalty': ['l2'],
             'C': c_range, 'class_weight': ['balanced', None]},
        ]
        if best:
            lr_params = [{'C': [0.3], 'class_weight': ['balanced'], 'max_iter': [1000], 'penalty': ['l2'], 'solver': ['liblinear']}]
        evaluate_model("Logistic Regression", LogisticRegression, lr_params, X_train, y_train, X_test, y_test, feature_selection=not correlated_features)

    # Linear SVM model
    if "LinSVM" in models:
        c_range = [100, 20, 10.0] + [i * 0.1 for i in range(1, 11)]
        params = [
            {'class_weight': ['balanced'], 'dual': [False], 'fit_intercept': [True], 'C': c_range}
        ]
        if best:
            params = [{'C': [0.4], 'class_weight': ['balanced'], 'dual': [False], 'fit_intercept': [True]}]
        evaluate_model("Linear SVM", LinearSVC, params, X_train, y_train, X_test, y_test,
                       feature_selection=not correlated_features)

    # Extreme Boost model
    if "XGBoost" in models:
        params = [
            {
                'subsample': [i * 0.1 for i in range(1, 11)],
                'n_estimators': [10, 50, 100, 200, 500],
                'max_depth': [i for i in range(1, 15)] + [None],
                'colsample_bytree': [i / len(X.columns) for i in range(1, 15)],
                'scale_pos_weight': [3.15],
                'tree_method': ['hist'],
            }
        ]
        if best:
            params = [{'colsample_bytree': [0.10638297872340426], 'n_estimators': [100], 'scale_pos_weight': [3.15], 'subsample': [1.0], 'tree_method': ['hist']}]
        evaluate_model("XGBoost", XGBClassifier, params, X_train_full, y_train, X_test_full, y_test,
                       feature_selection=False)

    # SVM model
    if "SVM" in models:
        SVM_params = [
            {"kernel": ["linear", "poly", "rbf", "sigmoid"]},
        ]
        if best:
            SVM_params = [{"kernel": ["rbf"]}]
        evaluate_model("SVM", SVC, SVM_params, X_train, y_train, X_test, y_test, feature_selection=not correlated_features)

    print("--- Finished in %s seconds --- \n" % int(time.time() - start_time))
    input("Press any key to quit:")


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for Training, Optimising and Evaluating models based on the clean data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-cf", "--correlated-features", action="store_true",
                        help="To use forward feature selection(slow) or to use correlated selection of features")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed for splitting the train-test sets")
    parser.add_argument("-b", "--best", action="store_true",
                        help="Do not search hyperparams, choose the best ones already predefined")
    parser.add_argument('models', nargs='*', type=str,
                        help="Space seperated list of models to train and evaluate, use all if empty, " +
                             "has to be in the set (Base, KNN, DecisionTree, RandomForest, GradBoost, XGBoost, " +
                             "LinearDA, LogisticReg, SVM, LinSVM)")
    args = parser.parse_args()
    config = vars(args)
    if not config['models']:
        config['models'] = ["Base", "KNN", "DecisionTree", "RandomForest", "GradBoost", "LinearDA", "LogisticReg", "SVM", "XGBoost", "LinSVM"]

    main(**config)
