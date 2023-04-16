import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
import seaborn as sns


def countMissingValues(df, name):
    print(f'============= missing values on {name} =============')
    print(df.isna().sum())
    print("total: ", len(df.index))

    print("==========================")
    print("")


def getMissingValueDistributions(df, name):
    print(f'============= distribution of missing values on {name} =============')
    print()
    print(df['native-country'].value_counts(normalize=True))
    print()
    print(df['workclass'].value_counts(normalize=True))
    print()
    print(df['occupation'].value_counts(normalize=True))
    print()
    print("==========================")
    print("")


def getBalanceTrainingData(df):
    positive_count = len(df[df['class'] == '>50K'])
    total = len(df.index)

    print("============= class imbalance on training data =============")

    print("positive count: ", positive_count)
    print("negative count: ", total-positive_count)
    print("total: ", total)
    print("ratio positive/total: ", positive_count/total)

    print("==========================")
    print("")


def main():
    existing_customers = pd.read_csv("data/existing-customers.csv")
    pot_customers = pd.read_csv("data/potential-customers.csv")

    print("=== Existing Customers ===")
    print(existing_customers.describe())
    print()
    print("=== Potential Customers ===")
    print(pot_customers.describe())
    print()

    getBalanceTrainingData(existing_customers)

    countMissingValues(existing_customers, "existing customers")
    countMissingValues(pot_customers, "potential customers")

    getMissingValueDistributions(existing_customers, "existing customers")
    getMissingValueDistributions(pot_customers, "existing customers")

    boxplot = existing_customers.boxplot(column=['capital-gain'])
    boxplot.plot()
    plot.show()

    #TODO view distribution of value frequencies between the two


if __name__ == '__main__':
    main()
