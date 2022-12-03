import argparse
import warnings

import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv
from benchmarks.alpa.general_model_case import read_general_model_case_tsv

show_name_dict = {
    "sr-greedy":   "Selective Replication (greedy)",
    "sr-ilp":      "Selective Replication (ilp)",

    "mp-ilp":      "Model Parallelism (ilp)",
    "mp-search":   "Model Parallelism (search)",
    "mp-greedy-2": "Pipeline Parallelism (#stage=2)",
    "mp-greedy-4": "Pipeline Parallelism (#stage=4)",
    "mp-greedy-8": "Pipeline Parallelism (#stage=8)",
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
    "sr-greedy", "sr-ilp",

    "mp-ilp", "mp-search",
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

    ax.set_ylim(bottom=75, top=max(y_max * 1.02, 102))
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


def plot_goodput_vs_num_devices(lines, threshold, show, folder):
    # Dict[policy -> Dict[num_devices -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_num_devices":
            continue

        policy, x, goodput = (
            line["policy_name"], line["num_devices"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, True, "#devices",
                        "Goodput vs. #devices", folder + "/goodput_vs_num_devices.png",
                        args.show)


def plot_goodput_vs_num_models(lines, threshold, show, folder):
    # Dict[policy -> Dict[num_models -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_num_models":
            continue

        policy, x, goodput = (
            line["policy_name"], line["num_models"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, False, "#models",
                        "Goodput vs. #models", folder + "/goodput_vs_num_models.png",
                        args.show)


def plot_goodput_vs_slo(lines, threshold, show, folder):
    # Dict[policy -> Dict[slo -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_slo":
            continue

        policy, x, goodput =  (
            line["policy_name"], line["slo_scale"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, True, "SLO Scale",
                        "Goodput vs. SLO Scale", folder + "/goodput_vs_slo.png",
                        args.show)


def plot_goodput_vs_rate_scale(lines, threshold, show, folder):
    # Dict[policy -> Dict[rate_scale -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_rate_scale":
            continue

        policy, x, goodput = (
            line["policy_name"], line["arrival_process_kwargs"]["rate_scale"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, False, "Rate Scale",
                        "Goodput vs. Rate Scale", folder + "/goodput_vs_rate_scale.png",
                        args.show)


def plot_goodput_vs_cv_scale(lines, threshold, show, folder):
    # Dict[policy -> Dict[cv_scale -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_cv_scale":
            continue

        policy, x, goodput = (
            line["policy_name"], line["arrival_process_kwargs"]["cv_scale"], line["goodput"])
        data[policy][x] = goodput

    plot_goodput_common(data, threshold, False, "CV Scale",
                        "Goodput vs. CV Scale", folder + "/goodput_vs_cv_scale.png",
                        args.show)


def plot_num_devices_vs_num_models(lines, threshold, show, folder):
    # Dict[policy -> Dict[cv_scale -> goodput]]
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for line in lines:
        if line["exp_name"] != "num_devices_vs_num_models":
            continue
        policy, y, x, goodput = (
            line["policy_name"],
            line["num_devices"],
            line["num_models"],
            line["goodput"]
        )
        raw_data[policy][x][y] = goodput

    goodput_goal = threshold

    output = "num_devices_vs_num_models"
    # sort the data, for each num_model, find out the min(num_device) that gives 99% goodput.
    data = defaultdict(lambda: defaultdict(dict))
    for policy in raw_data:
        for n_model in raw_data[policy]:
            min_device = 1e5
            for n_device in raw_data[policy][n_model]:
                goodput = raw_data[policy][n_model][n_device]
                if goodput >= goodput_goal:
                    if n_device < min_device:
                        min_device = n_device
            data[policy][n_model] = min_device

    print(data)
    if len(data) == 0:
        print(f"No data to draw for {output}. Skipped.")
        return

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
        ys = np.array(ys)
        curve = ax.plot(xs, ys, color=method2color(method), marker='*')
        curves.append(curve[0])
        legends.append(show_name(method))
        x_max = max(x_max, *xs)
        y_max = max(y_max, *ys)
    print(y_max)
    ax.set_ylim(bottom=0, top=y_max + 4)
    ax.set_ylabel("# devices")
    ax.set_xlabel("# models")
    ax.legend(curves, legends)
    ax.set_title("#devices vs. #models")

    if show:
        plt.show()

    fig.set_size_inches(figure_size)
    fig.savefig(folder + "/" + output, bbox_inches='tight')
    print(f"Output the plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--general-case", action="store_true")
    args = parser.parse_args()

    if args.general_case:
        lines = read_general_model_case_tsv(args.input)
    else:
        lines = read_equal_model_case_tsv(args.input)

    if len(args.input) > 1:
        folder = os.path.dirname(args.input)
    else:
        folder = ""

    threshold = 0.99

    plot_goodput_vs_num_devices(lines, threshold, args.show, folder)
    plot_goodput_vs_num_models(lines, threshold, args.show, folder)
    plot_goodput_vs_slo(lines, threshold, args.show, folder)
    plot_goodput_vs_rate_scale(lines, threshold, args.show, folder)
    plot_goodput_vs_cv_scale(lines, threshold, args.show, folder)
    plot_num_devices_vs_num_models(lines, threshold, args.show, folder)
