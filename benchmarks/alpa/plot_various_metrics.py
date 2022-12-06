import argparse
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv

show_name_dict = {
    "sr-greedy":   "Selective Replication (greedy)",
    "sr-search":   "Selective Replication (search)",
    "sr-ilp":      "Selective Replication (ilp)",

    "mp-ilp":         "Model Parallelism (ilp)",
    "mp-search":      "Model Parallelism (search)",
    "mp-search-100":  "Model Parallelism (search 100s)",
    "mp-search-1000": "Model Parallelism (search 1000s)",
    "mp-greedy-2":    "Pipeline Parallelism (#stage=2)",
    "mp-greedy-4":    "Pipeline Parallelism (#stage=4)",
    "mp-greedy-8":    "Pipeline Parallelism (#stage=8)",
}

def show_name(name):
    return show_name_dict.get(name, name)


method2color_dict = {
}

ct = 0
def method2color(name):
    global ct
    if name not in method2color_dict:
        method2color_dict[name] = f"C{ct}"
        ct += 1
    return method2color_dict[name]


method_order_list = [
    "sr-greedy", "sr-search", "sr-ilp",

    "mp-ilp", "mp-search",
    "mp-search-100", "mp-search-1000",
    "mp-greedy-2", "mp-greedy-4", "mp-greedy-8",
]

def method2order(name):
    return method_order_list.index(name)



def plot_goodput_common(data, threshold, increasing, xlabel, title, output, show):
    if len(data) == 0:
        print(f"No data to draw for {output}. Skipped.")
        return

    fig, ax = plt.subplots()
    figure_size = (5, 5)

    methods = list(data.keys())
    methods.sort(key=lambda x: method2order(x))

    curves = []
    legends = []
    first_good = []
    x_max = 0
    y_max = 0
    for method in methods:
        curve = data[method]
        xs, ys = zip(*curve.items())

        ys = np.array(ys) * 100
        curve = ax.plot(xs, ys, color=method2color(method), marker='*')
        curves.append(curve[0])
        legends.append(show_name(method))

        if increasing:
            iterator = range(len(xs))
        else:
            iterator = reversed(range(len(xs)))

        found = False
        for i in iterator:
            if ys[i] >= threshold * 100:
                first_good.append(xs[i])
                found = True
                break
        if not found:
            first_good.append(0)

        x_max = max(x_max, *xs)
        y_max = max(y_max, *ys)

    ax.set_ylim(bottom=85, top=max(y_max * 1.05, 100))
    ax.set_ylabel("Goodput (%)")
    ax.set_xlabel(xlabel)
    ax.legend(curves, legends)
    ax.set_title(title)

    for i in range(len(methods)):
        ax.axvline(first_good[i], color=method2color(methods[i]), linestyle=":")

    if show:
        plt.show()

    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")


def plot_goodput_vs_num_devices(lines, threshold, show):
    # Dict[policy -> Dict[num_devices -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_num_devices":
            continue

        policy, x, goodput = (
            line["policy_name"], line["num_devices"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, True, "#devices",
                        "Goodput vs. #devices", "goodput_vs_num_devices.png",
                        args.show)


def plot_goodput_vs_num_models(lines, threshold, show):
    # Dict[policy -> Dict[num_models -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_num_models":
            continue

        policy, x, goodput = (
            line["policy_name"], line["num_models"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, False, "#models",
                        "Goodput vs. #models", "goodput_vs_num_models.png",
                        args.show)


def plot_goodput_vs_slo(lines, threshold, show):
    # Dict[policy -> Dict[slo -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_slo":
            continue

        policy, x, goodput =  (
            line["policy_name"], line["slo_scale"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, True, "SLO (s)",
                        "Goodput vs. SLO", "goodput_vs_slo.png",
                        args.show)


def plot_goodput_vs_total_rate(lines, threshold, show):
    # Dict[policy -> Dict[total_rate -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_total_rate":
            continue

        policy, x, goodput = (
            line["policy_name"], line["total_rate"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, False, "Total rate (r/s)",
                        "Goodput vs. Total rate", "goodput_vs_total_rate.png",
                        args.show)


def plot_goodput_vs_cv(lines, threshold, show):
    # Dict[policy -> Dict[cv -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_cv":
            continue

        policy, x, goodput = (
            line["policy_name"], line["arrival_process_kwargs"]["cv"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, False, "CV",
                        "Goodput vs. CV", "goodput_vs_cv.png",
                        args.show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="res_various_metrics.tsv")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    lines = read_equal_model_case_tsv(args.input)

    threshold = 0.99

    plot_goodput_vs_num_devices(lines, threshold, args.show)
    plot_goodput_vs_num_models(lines, threshold, args.show)
    plot_goodput_vs_slo(lines, threshold, args.show)
    plot_goodput_vs_total_rate(lines, threshold, args.show)
    plot_goodput_vs_cv(lines, threshold, args.show)
