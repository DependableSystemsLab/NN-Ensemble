#! /usr/bin/python3

# Computes the Shannon equitability (EH) for measuring diversity across predicted labels
# 0 (all labels belong to a single class) <= EH <= 1 (labels equally distributed)

import sys
import pandas as pd
import statistics as stats
import math
import numpy as np


def get_eh(df):
    z = np.array(df, 'float')
    z = df / df.sum(axis=1).to_numpy()[:,np.newaxis]
    div = np.apply_along_axis(shannon, 1, z)
    return div


def shannon(y):
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    n = len(y)
    t = t[t!=0]
    H = -np.sum( t*np.log(t) )
    return H/np.log(n)


def get_avg_eh(df):
    temp_df = df.iloc[:,2:].apply(pd.Series.value_counts, axis=1)
    temp_df = temp_df.fillna(0)
    eh_arr = get_eh(temp_df)
    return stats.mean(eh_arr)


def get_corr_eh(df):
    temp_df = df.iloc[:,2:]

    temp_df2 = temp_df.mode(axis='columns', numeric_only=True)
    temp_df2["Mode"] = temp_df2.loc[:,0].astype(int)
    temp_df2["SingleMode"] = np.where(temp_df2[1].isnull(), True, False)
    temp_df2["CorrPlurality"] = np.where((temp_df2["Mode"] == df["test_label"]) & (temp_df2["SingleMode"]), True, False)

    temp_df2["CorrCount"] = temp_df.eq(df["test_label"], axis=0).sum(axis=1)
    num_classes = len(temp_df.columns)
    temp_df2["CorrMajority"] = np.where(temp_df2["CorrCount"] >= num_classes / 2., True, False)

    temp_df = pd.concat([df, temp_df2["CorrPlurality"], temp_df2["CorrMajority"]], axis=1)

    corr_plurality_df = (temp_df[temp_df["CorrPlurality"]]).iloc[:,:-2]
    inc_plurality_df = (temp_df[~temp_df["CorrPlurality"]]).iloc[:,:-2]
    corr_majority_df = (temp_df[temp_df["CorrMajority"]]).iloc[:,:-2]
    inc_majority_df = (temp_df[~temp_df["CorrMajority"]]).iloc[:,:-2]

    eh_corr_plurality = get_avg_eh(corr_plurality_df)
    eh_inc_plurality = get_avg_eh(inc_plurality_df)
    eh_corr_majority = get_avg_eh(corr_majority_df)
    eh_inc_majority = get_avg_eh(inc_majority_df)

    print("Corrrect Plurality EH =", eh_corr_plurality)
    print("Incorrect Plurality EH =", eh_inc_plurality)
    print("Correct Majority EH =", eh_corr_majority)
    print("Incorrect Majority EH =", eh_inc_majority)


def InvalidCommand():
    print("usage: python", sys.argv[0], "<csvfilename>")
    sys.exit(2)


def main(argv):
    if (len(argv) == 0):
        InvalidCommand()
    file_name = argv[0]

    df = pd.read_csv(file_name)

    eh = get_avg_eh(df)
    print("Shannon Equitability for", file_name, "is:", eh)

    get_corr_eh(df)


if __name__ == "__main__":
    main(sys.argv[1:])

