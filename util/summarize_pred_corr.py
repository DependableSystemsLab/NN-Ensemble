#! /usr/bin/python3

# Summarize the prediction and correct label files from multiple runs into a single file
# By taking the mode of the results

import sys
import pandas as pd
import numpy as np
import csv
import glob


def InvalidCommand():
    print("usage: python", sys.argv[0], "<csvfilename>")
    sys.exit(2)


def main(argv):
    if (len(argv) == 0):
        InvalidCommand()
    filename = argv[0]
    print("Filename : " + filename)

    corr_files = glob.glob("./analysis/" + filename + "-correct_pred-*.csv")
    print(corr_files)
    combine_files(filename, corr_files, "-correct_pred.csv")

    pred_files = glob.glob("./analysis/" + filename + "-predictions-*.csv")
    print(pred_files)
    combine_files(filename, pred_files, "-predictions.csv")


def combine_files(filename, filelist, analysis_suffix):
    filelist.sort()

    df = pd.read_csv(filelist[0])
    temp_df = pd.read_csv(filelist[0]).iloc[:,2]
    column_name = df.columns.values[-1]

    if (len(filelist) > 1):
        for file in filelist[1:]:
            f_df = pd.read_csv(file).iloc[:,2]
            temp_df = pd.concat([temp_df, f_df], axis=1)

        temp_df2 = temp_df.mode(axis='columns', numeric_only=True)
        temp_df2["rand"] = pd.Series([np.random.choice(i,1)[0] for i in (temp_df2.iloc[:,:2]).values])
        temp_df2[column_name] = np.where(temp_df2.iloc[:,1].notnull(), temp_df2["rand"], temp_df2.iloc[:,0]).astype(int)

        df = pd.concat([df.iloc[:,:2], temp_df2[column_name]], axis=1)
        df.to_csv("./analysis/pred_corr/" + filename + analysis_suffix, index=False)

    else:
        df.to_csv("./analysis/pred_corr/" + filename + analysis_suffix, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])

