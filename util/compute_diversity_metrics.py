#! /usr/bin/python3

import sys
import numpy as np
import pandas as pd
import statistics as stats
import math
import itertools as IT


def twoset(iterable):
    """twoset([A,B,C]) --> (A,B) (A,C) (B,C)"""
    s = list(iterable)
    return IT.chain.from_iterable(
        IT.combinations(s, r) for r in range(2,3))


def dict_twoset_counts(df):
    """twoset([A,B,C]) --> (A,B) (A,C) (B,C)"""
    """twoset_counts = {"A":{"B": [A==0 & B==0, A==0 & B==1, A==1 & B==0, A==1 & B==1],
                             "C": [A==0 & C==0, A==0 & C==1, A==1 & C==0, A==1 & C==1]},
                        "B":{"C": [B==0 & C==0, B==0 & C==1, B==1 & C==0, B==1 & C==1]}}"""
    result = {}
    for cols in twoset(df.columns):
        if not cols: continue
        result.setdefault(cols[0],{})
        for vals in IT.product([0,1], repeat=len(cols)):
            mask = np.logical_and.reduce([df[c]==v for c, v in zip(cols, vals)])
            cond = ' & '.join(['{}={}'.format(c,v) for c, v in zip(cols,vals)])
            n = len(df[mask])
            result[cols[0]].setdefault(cols[1],[]).append(n)
    return result


def get_diversity_stats(file_name):
    df = pd.read_csv(file_name)
    temp_df = df.iloc[:,2:-1]
    res = dict_twoset_counts(temp_df)

    col_list = list(df)[2:-1]
    two_pairs = list(IT.combinations(col_list, 2))

    L = len(two_pairs)
    dis = 0.
    doubledis = 0.
    qstat = 0.
    corr = 0.

    for mpair in two_pairs:
        m1 = mpair[0]
        m2 = mpair[1]
        n00 = res[m1][m2][0]
        n01 = res[m1][m2][1]
        n10 = res[m1][m2][2]
        n11 = res[m1][m2][3]

        dis += (n01 + n10) / (n11 + n10 + n01 + n00)
        doubledis += n00 / (n11 + n10 + n01 + n00)
        qstat += (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
        corr += (n11 * n00 - n01 * n10) / math.sqrt((n11 + n10)*(n01 * n00)*(n11 + n01)*(n10 * n00))

    dis /= L
    doubledis /= L
    qstat /= L
    corr /= L

    print("Disagreement measure:", dis)
    print("Double fault measure:", doubledis)
    print("Q statistic:", qstat)
    print("Correlation coefficient", corr)

def InvalidCommand():
    print("usage: python", sys.argv[0], "<csvfilename>")
    sys.exit(2)


def main(argv):
    if (len(argv) == 0):
        InvalidCommand()
    file_name = argv[0]

    print(file_name)
    get_diversity_stats(file_name)


if __name__ == "__main__":
    main(sys.argv[1:])

