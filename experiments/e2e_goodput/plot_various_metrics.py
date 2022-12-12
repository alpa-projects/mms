import argparse
import warnings

import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv
from benchmarks.alpa.general_model_case import read_general_model_case_tsv
from benchmarks.alpa.plot_various_metrics import plot_goodput_common, show_name, \
    method2color, method2order


def plot_goodput_vs_num_devices(lines, threshold, show, folder, pdf):
    # Dict[policy -> Dict[num_devices -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_num_devices":
            continue

        policy, x, goodput = (
            line["policy_name"], line["num_devices"], line["goodput"])
        data[policy][x] = goodput
    
    if pdf:
        output = os.path.join(folder, "goodput_vs_num_devices.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_num_devices.png")

    plot_goodput_common(data, threshold, True, "#devices",
                        "Workload satisfaction vs. #devices", output, args.show)


def plot_goodput_vs_num_models(lines, threshold, show, folder, pdf):
    # Dict[policy -> Dict[num_models -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_num_models":
            continue

        policy, x, goodput = (
            line["policy_name"], line["num_models"], line["goodput"])
        data[policy][x] = goodput
    
    if pdf:
        output = os.path.join(folder, "goodput_vs_num_models.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_num_models.png")

    plot_goodput_common(data, threshold, False, "#models",
                        "Workload satisfaction vs. #models", output, args.show)


def plot_goodput_vs_slo(lines, threshold, show, folder, pdf):
    # Dict[policy -> Dict[slo -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_slo":
            continue

        policy, x, goodput =  (
            line["policy_name"], line["slo_scale"], line["goodput"])
        data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_slo.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_slo.png")

    plot_goodput_common(data, threshold, True, "SLO Scale",
                        "Workload satisfaction vs. SLO Scale", output, args.show)


def plot_goodput_vs_rate(lines, threshold, show, folder, pdf):
    # Dict[policy -> Dict[rate -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_rate":
            continue

        policy, x, goodput = (
            line["policy_name"], line["total_rate"], line["goodput"])
        data[policy][x] = goodput
    
    if pdf:
        output = os.path.join(folder, "goodput_vs_rate.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_rate.png")

    plot_goodput_common(data, threshold, False, "Rate(r/s)",
                        "Workload satisfaction vs. Rate", output, args.show)


def plot_goodput_vs_rate_scale(lines, threshold, show, folder, pdf):
    # Dict[policy -> Dict[rate_scale -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_rate_scale":
            continue

        policy, x, goodput = (
            line["policy_name"], line["arrival_process_kwargs"]["rate_scale"], line["goodput"])
        data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_rate_scale.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_rate_scale.png")

    plot_goodput_common(data, threshold, False, "Rate Scale",
                        "Workload satisfaction vs. Rate Scale", output, args.show)


def plot_goodput_vs_cv(lines, threshold, show, folder, pdf):
    # Dict[policy -> Dict[cv -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_cv":
            continue

        policy, x, goodput = (
            line["policy_name"], line["arrival_process_kwargs"]["cv"], line["goodput"])
        data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_cv.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_cv.png")


    plot_goodput_common(data, threshold, False, "CV",
                        "Workload satisfaction vs. CV", output, args.show)



def plot_goodput_vs_cv_scale(lines, threshold, show, folder, pdf):
    # Dict[policy -> Dict[cv_scale -> goodput]]
    data = defaultdict(lambda: defaultdict(dict))

    for line in lines:
        if line["exp_name"] != "goodput_vs_cv_scale":
            continue

        policy, x, goodput = (
            line["policy_name"], line["arrival_process_kwargs"]["cv_scale"], line["goodput"])
        data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_cv_scale.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_cv_scale.png")

    plot_goodput_common(data, threshold, False, "CV Scale",
                        "Workload satisfaction vs. CV Scale", output, args.show)


def plot_num_devices_vs_num_models(lines, threshold, show, folder, pdf):
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
            min_device = 1e3
            for n_device in raw_data[policy][n_model]:
                goodput = raw_data[policy][n_model][n_device]
                if goodput >= goodput_goal:
                    if n_device < min_device:
                        min_device = n_device
            data[policy][n_model] = min_device
            if min_device == 1e3:
                data[policy][n_model] = n_model // 2

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

    if pdf:
        output += ".pdf"
    else:
        output += ".png"

    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--general-case", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    if len(args.input) > 1:
        folder = os.path.dirname(args.input)
    else:
        folder = ""

    threshold = 0.99

    if args.general_case:
        lines = read_general_model_case_tsv(args.input)
    else:
        lines = read_equal_model_case_tsv(args.input)

    plot_goodput_vs_num_devices(lines, threshold, args.show, folder, args.pdf)
    plot_goodput_vs_num_models(lines, threshold, args.show, folder, args.pdf)
    plot_goodput_vs_slo(lines, threshold, args.show, folder, args.pdf)
    if args.synthetic:
        plot_goodput_vs_rate(lines, threshold, args.show, folder, args.pdf)
        plot_goodput_vs_cv(lines, threshold, args.show, folder, args.pdf)
    else:
        plot_goodput_vs_rate_scale(lines, threshold, args.show, folder, args.pdf)
        plot_goodput_vs_cv_scale(lines, threshold, args.show, folder, args.pdf)
    plot_num_devices_vs_num_models(lines, threshold, args.show, folder, args.pdf)
