import glob
import pandas as pd


def sort_files(benchmark, mode):
    files = glob.glob("./analysis/*" + benchmark + "*-" + mode +".csv")
    categories = find_categories(files)
    grouped_files = {}
    for cat in categories:
        filter_files = glob.glob("./analysis/*" + benchmark + "-" + cat + "-" + mode + ".csv")
        filter_files.sort()
        grouped_files[cat] = filter_files
    return grouped_files


def find_categories(files):
    results = set()
    for f in files:
        fsplit = f.split('-')
        if (len(fsplit) != 3):
            continue
        results.add(fsplit[1])
    return list(results)


def col_combine(filegroup):
    init_df = pd.read_csv(filegroup[0])
    init_df = init_df.iloc[:,:-1]
    for f in filegroup:
        df = pd.read_csv(f)
        df = df.iloc[:,-1:]
        init_df = pd.concat([init_df, df], axis=1)
    return init_df


def add_col_sum(df):
    df["sum"] = df.iloc[:,2:].sum(axis=1)
    return df


def write_combined_file(df, cat, benchmark, mode):
    df.to_csv("./analysis/summary/" + benchmark + "-" + cat +"-" + mode + ".csv", index=False)


def main():
    benchmarks = ["mnist", "cifar10", "cifarhundred"]
    for b in benchmarks:

        grouped_pred = sort_files(b, "predictions")
        for cat, filegroup in grouped_pred.items():
            df = col_combine(filegroup)
            write_combined_file(df, cat, b, "predictions")

        grouped_corr = sort_files(b, "correct_pred")
        for cat, filegroup in grouped_corr.items():
            df = col_combine(filegroup)
            df = add_col_sum(df)
            write_combined_file(df, cat, b, "correct_pred")


if __name__ == "__main__":
    main()

