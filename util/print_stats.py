#! /usr/bin/python3

import sys
import csv
import statistics as stats
import math
from datetime import datetime

def print_stats(samples, sample_name):
    average = stats.mean(samples)
    std = stats.stdev(samples)
    n = len(samples)
    z = 1.96
    margin_err = z * std / math.sqrt(n)
    average = round(average*100, 5)
    margin_err = round(margin_err*100, 5)

    print("Average", sample_name,":", average, "% +/-", margin_err, "%")
    return average, margin_err


def load_vals(file_name):
    acc_vals = []
    sdc_vals = []
    with open(file_name, "r") as r_file:
        reader = csv.reader(r_file)
        for row in reader:
            for col_id, col_val in enumerate(row):
                col_val = float(col_val)
                if (col_id == 0):
                    acc_vals.append(col_val)
                else:
                    sdc_vals.append(col_val)
        return acc_vals, sdc_vals


def InvalidCommand():
    print("usage: python", sys.argv[0], "<csvfilename>")
    sys.exit(2)

def save_to_file(file_name, sample_size, acc_avg, acc_err, sdc_avg, sdc_err):
    datetimenow = datetime.now()
    summary_csv = "./analysis/acc_sdc_stats/summary.csv"

    with open(summary_csv, "a") as a_file:
        csvwriter = csv.writer(a_file, delimiter=',',
                               quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([datetimenow, file_name, sample_size, acc_avg, acc_err, sdc_avg, sdc_err])

def main(argv):
    if (len(argv) == 0):
        InvalidCommand()
    file_name = argv[0]

    print("Stats for", file_name)
    acc_vals, sdc_vals = load_vals(file_name)
    acc_avg, acc_err = print_stats(acc_vals, "Accuracy")
    sdc_avg, sdc_err = print_stats(sdc_vals, "SDC Rate")
    sample_size = len(acc_vals)
    save_to_file(file_name, sample_size, acc_avg, acc_err, sdc_avg, sdc_err)

if __name__ == "__main__":
    main(sys.argv[1:])

