import pandas as pd
import pickle
from contextlib import redirect_stdout
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from Step2_train_evaluate import EVPC_score, cost_extracted_score

class Logger:
    def __init__(self, file):
        self.f = file

    def log(self, string=""):
        print(string)
        with redirect_stdout(self.f):
            print(string)


def main():
    ### Chosen parameters and features ###
    dataset = pd.read_csv("data/existing-customers-CLEAN.csv", index_col=0)
    datasetpot = pd.read_csv("data/potential-customers-CLEAN.csv", index_col=0)

    # Hist Gradient boost
    params = { 'class_weight': 'balanced', 'max_depth': 4, 'max_iter': 500,
              'categorical_features': [col for col in datasetpot if col not in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]}

    ### my student number as random state for better reproducibility ###
    state = 20191842

    # split X and Y
    X = dataset.loc[:, dataset.columns != 'class']
    y = dataset.loc[:, dataset.columns == 'class']

    # Split train and test set, test set never used in training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, shuffle=True, random_state=state)

    print("Training Model...\n")
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    logFile = open('output/logtemp.txt', 'w')
    l = Logger(logFile)

    l.log("=== Evaluating Model === \n")

    y_test_predictions = model.predict(X_test)

    perms = permutation_importance(model, X_train, y_train)
    plt.figure(figsize=[20, 8])
    plt.barh(model.feature_names_in_, perms['importances_mean'])
    plt.xlabel('importance')
    plt.title("Feature Permutation Importance")
    plt.show()

    l.log(f"F1: {f1_score(y_test, y_test_predictions)}")
    l.log(f"AUC ROC score: {roc_auc_score(y_test, y_test_predictions)}")
    l.log(f"precision: {precision_score(y_test, y_test_predictions)}")
    l.log(f"recall: {recall_score(y_test, y_test_predictions)}")
    l.log(f"accuracy: {accuracy_score(y_test, y_test_predictions)}")
    l.log(f"ratio >50k to total: {float(y_test_predictions.sum()) / len(y_test_predictions)}")
    l.log(f"real ratio >50k to total: {float(y_test.sum()) / len(y_test)}")
    l.log()
    l.log(f"extracted value coefficient: {cost_extracted_score(y_test, y_test_predictions)}")
    evpc = EVPC_score(y_test, y_test_predictions)
    l.log(f"EVPC score: {evpc}")
    l.log()
    l.log(confusion_matrix(y_test, y_test_predictions, labels=[True, False]))
    l.log()
    l.log("========\n")

    l.log("=== Predicting Potential Customers ===\n")

    y_predictions = model.predict(datasetpot)
    l.log(f">50k predictions: {y_predictions.sum()}")
    l.log(f"Ratio >50k to total: {float(y_predictions.sum()) / len(y_predictions)}")
    l.log()
    l.log(f"Expected Gain: {-1 * y_predictions.sum() * evpc}")
    l.log()
    l.log("======\n")

    userinput = input("Save the results?(y/n):")
    logFile.close()
    if userinput == "y":
        # save log
        os.remove("output/log.txt")
        os.rename('output/logtemp.txt', 'output/log.txt')

        # save model
        mFile = open("output/model.pickle", "wb")
        pickle.dump(model, mFile)
        mFile.close()

        # save predictions as csv
        output_predictions = pd.read_csv("data/potential-customers.csv", index_col=0)
        output_predictions['>50K'] = y_predictions

        output_predictions.to_csv("output/potential-customers-predictions.csv")

        # save selected customers as txt
        predicted_rows = output_predictions.index[output_predictions['>50K'] == True].tolist()

        with open('output/selected.txt', 'w') as f:
            for i, r in enumerate(predicted_rows):
                if i != 0:
                    f.write('\n')
                f.write(r)

            f.close()
            print("Results saved, goodbye!")
    else:
        os.remove("output/logtemp.txt")
        print("Not saved, goodbye!")


if __name__ == '__main__':
    main()
