#! /usr/bin/python3

# Computes the Gini coefficient for measuring distribution of correct classifications among classifiers
# 0 (no dominant combinations of classifiers) <= Gini <= 1 (same group of classifiers are always correct)

import sys
import pandas as pd
import statistics as stats
import math


def gini_coefficient(arr):
    total_abs_diff = 0
    mean = stats.mean(arr)
    n = len(arr)

    for i in range(n):
        for j in range(n):
            total_abs_diff += abs(arr[i] - arr[j])

    return total_abs_diff / (2. * n * n * mean)


def get_counts(file_name):
    df = pd.read_csv(file_name)
    col_list = list(df)[2:-1]
    dfg = df.groupby(col_list).size()
    dfg = dfg.tolist()
    return dfg


def InvalidCommand():
    print("usage: python", sys.argv[0], "<csvfilename>")
    sys.exit(2)


def fill_zeros(a, maxlen):
    a = a + [0]*(maxlen - len(a))
    return a


def main(argv):
    if (len(argv) == 0):
        InvalidCommand()
    file_name = argv[0]

    count_arr = get_counts(file_name)
    count_arr = fill_zeros(count_arr, 128)

    gc = gini_coefficient(count_arr)
    print("Gini coefficient for", file_name, "is:", gc)


if __name__ == "__main__":
    main(sys.argv[1:])

