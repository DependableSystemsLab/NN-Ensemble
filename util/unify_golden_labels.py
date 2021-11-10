import sys
import getopt
import json
import glob


def unify(benchmark):
    files = glob.glob("./golden_labels/*_" + benchmark + "-golden-labels.txt")
    inter = None
    print(files)

    for f in files:
        with open(f, "r") as r_file:
            labels = json.load(r_file)
            if inter is None:
                inter = set(labels)
            else:
                inter = inter.intersection(set(labels))

    print("Size of unified set:", str(len(inter)))
    interlist = list(inter)
    with open("./golden_labels/unified-" + benchmark + "-golden.txt", "w") as w_file:
        json.dump(interlist, w_file)


def InvalidCommand():
    print("usage: python", sys.argv[0], "-b <benchmark>")
    sys.exit(2)


def main(argv):
    benchmark = ''
    if (len(argv) == 0):
        InvalidCommand()
    try:
        opts, args = getopt.getopt(argv, "h:b:", [])
    except getopt.GetoptError:
        InvalidCommand()
    for opt, arg in opts:
        if opt == '-h':
            InvalidCommand()
        elif opt in ('-b'):
            benchmark = arg
    print("Benchmark chosen is", benchmark)
    unify(benchmark)


if __name__ == "__main__":
    main(sys.argv[1:])

