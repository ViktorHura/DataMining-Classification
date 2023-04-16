import pandas as pd
import numpy as np

import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


from Step0_EDA import countMissingValues


def oneHotEncode(ex, pot, column):
    encoder = OneHotEncoder(sparse=False)
    # fit encoder to existing
    encoder = encoder.fit(ex[column].values.reshape(-1, 1))

    out = []

    # transform existing and potential
    for df in [ex, pot]:
        encoded = encoder.transform(df[column].values.reshape(-1, 1))

        feature_names = encoder.get_feature_names_out(input_features=[column])
        encoded = pd.DataFrame(encoded, columns=feature_names)

        encoded.index = df.index
        df = pd.concat([df, encoded], axis=1)
        df.drop(columns=[column], inplace=True)
        out.append(df)

    return out


def scalingNormalization(ex, pot, column):
    scaler = MinMaxScaler()
    # fit encoder to existing
    scaler = scaler.fit(ex[column].values.reshape(-1, 1))

    out = []

    # transform existing and potential
    for df in [ex, pot]:
        scaled = scaler.transform(df[column].values.reshape(-1, 1))

        feature_names = scaler.get_feature_names_out(input_features=[column])
        scaled = pd.DataFrame(scaled, columns=feature_names)

        scaled.index = df.index
        df[column] = scaled[column]
        out.append(df)

    return out


def encodeAndScale(ex, pot):
    ex["class"] = ex["class"].map({'>50K': True, '<=50K': False})

    ex.drop(columns=['education'], inplace=True)
    pot.drop(columns=['education'], inplace=True)

    ex.drop(columns=['native-country'], inplace=True)
    pot.drop(columns=['native-country'], inplace=True)

    oneHotEncodeColumns = ['marital-status', 'relationship', 'race', 'sex']
    scaleNormalizeColumns = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    for c in oneHotEncodeColumns:
        ex, pot = oneHotEncode(ex, pot, c)

    for c in scaleNormalizeColumns:
        ex, pot = scalingNormalization(ex, pot, c)

    return ex, pot


def removeUnusualCapitalGain(df, name):
    found = df[df['capital-gain'] == 99999].shape[0]
    if found:
        print(f"Found {found} unusual values in the capital-gain column of {name}")
        df['capital-gain'].replace(99999, np.median(df['capital-gain'].values), inplace=True)



def imputeWorkclassOccupation(ex, pot):
    # label encode for purpose of imputation
    workclass_labels = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    occupation_labels = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]

    rtn = []

    for df in [ex, pot]:
        df['workclass'] = df['workclass'].apply(lambda x: workclass_labels.index(x) if x in workclass_labels else x)
        df['occupation'] = df['occupation'].apply(lambda x: occupation_labels.index(x) if x in occupation_labels else x)
        rtn.append(df)

    ex, pot = rtn

    # missForest label

    # train imputer on existing
    imputer = MissForest(max_features='sqrt')
    imputer = imputer.fit(ex.drop(columns=['class']), cat_vars=[1,3])


    exIm = imputer.transform(ex.drop(columns=['class']))
    potIm = imputer.transform(pot)

    # add column names back
    exIm = pd.DataFrame(exIm, index=ex.index, columns=ex.drop(columns=['class']).columns)
    exIm['class'] = ex['class']
    potIm = pd.DataFrame(potIm, index=pot.index, columns=pot.columns)

    # label decode
    rtn = []
    for df in [exIm, potIm]:
        df['workclass'] = df['workclass'].apply(lambda x: workclass_labels[int(x)])
        df['occupation'] = df['occupation'].apply(lambda x: occupation_labels[int(x)])
        rtn.append(df)
    exIm, potIm = rtn

    # oneHotEncode the inputed values
    for c in ['workclass', 'occupation']:
        exIm, potIm = oneHotEncode(exIm, potIm, c)

    return exIm, potIm


def main():
    existing_customers = pd.read_csv("data/existing-customers.csv", index_col=0)
    pot_customers = pd.read_csv("data/potential-customers.csv", index_col=0)

    print("removing unusual capital gain values")
    removeUnusualCapitalGain(existing_customers, "existing customers")
    removeUnusualCapitalGain(pot_customers, "potential customers")

    existing_customers, pot_customers = encodeAndScale(existing_customers, pot_customers)

    existing_customers, pot_customers = imputeWorkclassOccupation(existing_customers, pot_customers)

    countMissingValues(existing_customers, "existing")
    countMissingValues(pot_customers, "potential")

    existing_customers.to_csv("data/existing-customers-CLEAN.csv")
    pot_customers.to_csv("data/potential-customers-CLEAN.csv")


if __name__ == '__main__':
    main()
