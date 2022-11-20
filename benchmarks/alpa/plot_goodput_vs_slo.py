import argparse
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv

show_name_dict = {
    "sr-greedy":   "Selective Replication (greedy)",
    "sr-ilp":      "Selective Replication (ilp)",

    "mp-ilp":      "Model Parallelism (ilp)",
    "mp-search":   "Model Parallelism (search)",
    "mp-greedy-2": "Pipeline Parallelism (#stage=2)",
    "mp-greedy-4": "Pipeline Parallelism (#stage=4)",
    "mp-greedy-8": "Pipeline Parallelism (#stage=8)",
    "sr-greedy-batch":   "Selective Replication (greedy) Batching",
    "mp-search-batch":   "Model Parallelism (search) Batching",
    "mp-greedy-8-batch": "Pipeline Parallelism (#stage=8) Batching",
}

def show_name(name):
    return show_name_dict.get(name, name)


method2color_dict = {
}

ct = 0
def method2color(name):
    global ct
    name = name[:-6] if "batch" in name else name
    if name not in method2color_dict:
        method2color_dict[name] = f"C{ct}"
        ct += 1
    return method2color_dict[name]


method_order_list = [
    "sr-greedy", "sr-ilp",

    "mp-ilp", "mp-search",
    "mp-greedy-2", "mp-greedy-4", "mp-greedy-8",
    "sr-greedy-batch", "mp-search-batch", "mp-greedy-8-batch"
]

def method2order(name):
    return method_order_list.index(name)


def read_data(filename):
    # Dict[policy -> Dict[slo -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    rate = cv = None

    for line in read_equal_model_case_tsv(filename):
        policy, slo, goodput, total_rate, kwargs = (
            line["policy_name"], line["slo"], line["goodput"],
            line["total_rate"], line["arrival_process_kwargs"])

        if rate is None:
            rate = total_rate
            cv = kwargs["cv"]

        data[policy][slo] = goodput

    return data, {"total_rate": rate, "per_model_cv": cv}


def plot_goodput_vs_slo(data, title, output, show):
    fig, ax = plt.subplots()
    figure_size = (5, 5)

    methods = list(data.keys())
    methods.sort(key=lambda x: method2order(x))

    curves = []
    legends = []
    x_max = 0
    y_max = 0
    for method in methods:
        curve = data[method]
        xs, ys = zip(*curve.items())
        ys = np.array(ys) * 100
        if "batch" in method:
            curve = ax.plot(xs, ys, "--*", color=method2color(method))
        else:
            curve = ax.plot(xs, ys, "-*", color=method2color(method))
        curves.append(curve[0])
        legends.append(show_name(method))

        x_max = max(x_max, *xs)
        y_max = max(y_max, *ys)

    ax.set_ylim(bottom=0, top=max(y_max * 1.05, 100))
    ax.set_ylabel("Goodput (%)")
    ax.set_xlabel("SLO (second)")
    ax.set_xscale("log")
    xticks = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xticks([], minor=True)
    ax.legend(curves, legends)
    ax.set_title(title)

    if show:
        plt.show()

    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="res_goodput_vs_slo.tsv")
    parser.add_argument("--output", type=str, default="goodput_vs_slo.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data, params = read_data(args.input)
    title = ", ".join(f"{k} = {v}" for k, v in params.items())
    plot_goodput_vs_slo(data, title, args.output, args.show)
