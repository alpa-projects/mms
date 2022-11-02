import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


def read_data(filename):
    # Dict[policy -> Dict[slo -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in open(filename):
        exp_name, policy, slo, goodput = line.split("\t")
        data[policy][slo] = goodput

    return data


def plot_goodput_vs_slo(data):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="res_goodput.tsv")
    parser.add_argument("--output", type=str, default="goodput.png")
    args = parser.parse_args()

    data = read_data(args.input)
    plot_goodput_vs_slo(data)
